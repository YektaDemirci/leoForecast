# Use this

import os
import sys
import csv
import gzip
from datetime import datetime
from collections import defaultdict
import glob

# Increase the CSV field size limit to handle potentially corrupted files
csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# Format detection & helpers
# ---------------------------------------------------------------------------

# Sanity threshold — any timestamp before this is treated as corrupt/malformed.
# Early 2000s data: nothing valid should be before year 2000.
MIN_TS = 946684800  # 2000-01-01 00:00:00 UTC


def detect_format(first_line):
    """Return 'v1' or 'v3' based on the first line of the file."""
    if first_line.startswith('#V3.'):
        return 'v3'
    if first_line.startswith('V1.0'):
        return 'v1'
    return 'v1'  # default fallback


def open_file(filepath):
    """Open a plain or gzipped text file, returning a text-mode file object."""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt', encoding='utf-8', errors='replace')
    return open(filepath, 'r', errors='replace')


# ---------------------------------------------------------------------------
# Per-format parsers
# ---------------------------------------------------------------------------

def parse_v1(filepath):
    """
    Parse old-format .snmp files (V1.0).
    Single row type — CSV with a header row.
    Key columns: timestamp, awcTpFdbAddress, awcTpFdbDestOctetsImmed
    ap_key = filepath (no AP column; each file is one AP).
    Returns list of (ts, ap_key, mac, dest_octets).
    """
    records = []
    try:
        with open_file(filepath) as f:
            first_line = f.readline().strip()
            if not first_line.startswith('V1.0'):
                f.seek(0)

            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts          = int(row['timestamp'])
                    if ts < MIN_TS:
                        continue
                    mac         = row['awcTpFdbAddress'].strip()
                    dest_octets = int(row['awcTpFdbDestOctetsImmed']) if row.get('awcTpFdbDestOctetsImmed') else 0
                    records.append((ts, filepath, mac, dest_octets))
                except (KeyError, ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
    return records


def parse_v3(filepath):
    """
    Parse new-format .snmp.gz files (V3.x).
    Mixed row types per line:
      #c1  — device state header  (we skip these data rows)
      #c2  — traffic counters header  (THIS is what we want)
      c2   — traffic data rows

    #c2 columns: timestamp, AP, awcTpFdbAddress, awcTpFdbClassID,
                 awcTpFdbSrcOctetsImmed, awcTpFdbDestOctetsImmed, ...

    ap_key = AP column value (more reliable than filepath for roaming detection).
    Returns list of (ts, ap_key, mac, dest_octets).
    """
    c2_columns = None
    records = []

    try:
        with open_file(filepath) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # Capture the #c2 header line to learn column positions
                if line.startswith('#c2,'):
                    c2_columns = line[1:].split(',')  # strip leading '#'
                    continue

                # Skip all other comment/header lines (#V3.x, #sys, #if, #c1)
                if line.startswith('#'):
                    continue

                # Only process c2 data rows
                row_type = line.split(',', 1)[0]
                if row_type != 'c2':
                    continue

                if c2_columns is None:
                    continue  # header not seen yet

                values = line.split(',')
                # Pad in case of missing trailing fields
                if len(values) < len(c2_columns):
                    values += [''] * (len(c2_columns) - len(values))

                row = dict(zip(c2_columns, values))

                try:
                    ts          = int(row['timestamp'])
                    if ts < MIN_TS:
                        continue
                    ap_key      = row['AP'].strip()
                    mac         = row['awcTpFdbAddress'].strip()
                    dest_octets = int(row['awcTpFdbDestOctetsImmed']) if row.get('awcTpFdbDestOctetsImmed') else 0
                    records.append((ts, ap_key, mac, dest_octets))
                except (KeyError, ValueError, TypeError):
                    continue

    except Exception as e:
        print(f"  Error reading {filepath}: {e}")

    return records


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def process_snmp_files(base_dirs, output_file, interval_seconds=600):
    output_file = f'{interval_seconds}_V3_' + output_file

    # time_bucket -> total bits in interval
    time_aggregates = defaultdict(int)

    # time_bucket -> set of active MAC addresses
    active_users = defaultdict(set)

    # (ap_key, mac) -> last counter value
    # Scoped per AP so roaming devices don't generate spurious deltas:
    # when a device roams its counter resets on the new AP, which would
    # otherwise be misread as a 32-bit wrap (~4 GB spurious spike).
    last_counters   = {}
    last_timestamps = {}

    # Discover all .snmp and *_snmp.gz files under base_dirs
    filepaths = []
    for base_dir in base_dirs:
        patterns = [
            ('*.snmp',    False),   # v1 plain
            ('*.snmp.gz', False),   # v3 dot-extension variant
            ('*_snmp.gz', False),   # v3 underscore-extension variant (e.g. AcadBldg1AP3_snmp.gz)
        ]
        for pattern, _ in patterns:
            found = glob.glob(os.path.join(base_dir, '**', pattern), recursive=True)
            print(f"  {base_dir}  /  {pattern}  →  {len(found)} files")
            filepaths.extend(found)

    # Deduplicate in case patterns overlap
    filepaths = list(set(filepaths))

    print(f"Found {len(filepaths)} SNMP files. Reading...")

    # Read all records across all files
    records = []
    for filepath in filepaths:
        try:
            with open_file(filepath) as f:
                first_line = f.readline().strip()
            fmt = detect_format(first_line)
        except Exception as e:
            print(f"  Could not open {filepath}: {e}")
            continue

        if fmt == 'v3':
            parsed = parse_v3(filepath)
        else:
            parsed = parse_v1(filepath)

        records.extend(parsed)

    print(f"Read {len(records):,} records. Sorting...")
    records.sort(key=lambda x: x[0])

    print("Aggregating...")
    for ts, ap_key, mac, dest_octets in records:
        bucket_ts = (ts // interval_seconds) * interval_seconds

        # Per-AP, per-MAC key prevents cross-AP counter pollution
        key = (ap_key, mac)

        if key in last_counters:
            last_val = last_counters[key]
            last_ts  = last_timestamps[key]

            # Compute delta — handle 32-bit counter wrap within the same AP
            if dest_octets >= last_val:
                delta = dest_octets - last_val
            else:
                delta = (2**32 - last_val) + dest_octets

            total_duration = ts - last_ts
            start_bucket   = (last_ts // interval_seconds) * interval_seconds

            if total_duration == 0 or start_bucket == bucket_ts:
                # Entire delta falls in one bucket
                time_aggregates[bucket_ts] += delta
                if delta > 0:
                    active_users[bucket_ts].add(mac)
            else:
                # Distribute proportionally based on actual time overlap per bucket
                for b_ts in range(start_bucket, bucket_ts + interval_seconds, interval_seconds):
                    overlap = min(ts, b_ts + interval_seconds) - max(last_ts, b_ts)
                    if overlap > 0:
                        contribution = round(delta * overlap / total_duration)
                        time_aggregates[b_ts] += contribution
                        if contribution > 0:
                            active_users[b_ts].add(mac)
        else:
            # First sighting of this MAC on this AP — begin tracking, no delta yet
            active_users[bucket_ts].add(mac)

        last_counters[key]   = dest_octets
        last_timestamps[key] = ts

    # Write output — multiply bytes × 8 to convert to bits
    print(f"Writing {output_file} ...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['date', 'OT', 'Active_Users'])

        for bucket_ts in sorted(time_aggregates.keys()):
            dt_str = datetime.fromtimestamp(bucket_ts).strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([
                dt_str,
                time_aggregates[bucket_ts] * 8,
                len(active_users[bucket_ts])
            ])

    print(f"Done. {len(time_aggregates):,} buckets written to {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- v1 data (old format, plain .snmp under agentnews / klebb) ----------
    # base_directories = ['agentnews', 'klebb']

    # --- v3 data (new format, gzipped .snmp.gz under date-stamped folders) --
    # base_directories = ['031101', '031102', '031103']

    # --- mixed: pass both sets together -------------------------------------
    base_directories = ['agentnews', 'klebb', 'boi']

    process_snmp_files(base_directories, 'aggregated_output.tsv', interval_seconds=600)
