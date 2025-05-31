import csv
import serial
import time
import serial.tools.list_ports
import platform
from datetime import datetime

CSV_FILE = 'plates_log.csv'
RATE_PER_MINUTE = 5  # Amount charged per minute

def detect_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    system = platform.system()
    for port in ports:
        desc = port.description.lower()
        if system == "Windows" and "com12" in port.device.lower():
            return port.device
        elif system == "Linux" and ("ttyusb" in port.device.lower() or "ttyacm" in port.device.lower()):
            return port.device
        elif system == "Darwin" and ("usbmodem" in port.device.lower() or "usbserial" in port.device.lower()):
            return port.device
    return None

def parse_arduino_data(line):
    try:
        parts = line.strip().split(',')
        print(f"[ARDUINO] Parsed parts: {parts}")
        if len(parts) != 2:
            return None, None
        plate = parts[0].strip().upper()
        balance_str = ''.join(c for c in parts[1] if c.isdigit())
        print(f"[ARDUINO] Cleaned balance: {balance_str}")
        if balance_str:
            balance = int(balance_str)
            return plate, balance
        return None, None
    except ValueError as e:
        print(f"[ERROR] Value error in parsing: {e}")
        return None, None

def process_payment(plate, balance, ser):
    try:
        with open(CSV_FILE, 'r') as f:
            rows = list(csv.reader(f))
            print(f"[DEBUG] CSV Contents: {rows}")
        if not rows:
            print("[ERROR] CSV is empty")
            return

        header = ['Plate Number', 'Payment Status', 'Timestamp']
        if rows[0] != header:
            print("[ERROR] CSV header missing or incorrect. Expected:", header)
            rows.insert(0, header)
        entries = rows[1:] if len(rows) > 1 else []
        found = False

        for i, row in enumerate(entries):
            print(f"[DEBUG] Checking row: {row}")
            if row[0].strip() == plate and row[1].strip() == '0':
                found = True
                try:
                    entry_time_str = row[2]
                    entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
                    exit_time = datetime.now()
                    minutes_spent = int((exit_time - entry_time).total_seconds() / 60) + 1
                    amount_due = minutes_spent * RATE_PER_MINUTE

                    while len(entries[i]) < 4:
                        entries[i].append('')
                    entries[i][3] = exit_time.strftime('%Y-%m-%d %H:%M:%S')

                    if balance < amount_due:
                        print(f"[PAYMENT] Insufficient balance: {balance} < {amount_due}")
                        ser.write(b'I\n')
                        return
                    else:
                        new_balance = balance - amount_due
                        print("[WAIT] Waiting for Arduino to be READY...")
                        start_time = time.time()
                        while True:
                            if ser.in_waiting:
                                arduino_response = ser.readline().decode().strip()
                                print(f"[ARDUINO] {arduino_response}")
                                if arduino_response == "READY":
                                    break
                            if time.time() - start_time > 10:
                                print("[ERROR] Timeout waiting for Arduino READY")
                                return
                            time.sleep(0.1)

                        ser.write(f"{new_balance}\r\n".encode())
                        print(f"[PAYMENT] Sent new balance {new_balance}")

                        start_time = time.time()
                        print("[WAIT] Waiting for Arduino confirmation...")
                        while True:
                            if ser.in_waiting:
                                confirm = ser.readline().decode().strip()
                                print(f"[ARDUINO] {confirm}")
                                if "DONE" in confirm:
                                    print("[ARDUINO] Write confirmed")
                                    entries[i][1] = '1'
                                    break
                            if time.time() - start_time > 10:
                                print("[ERROR] Timeout waiting for confirmation")
                                break
                            time.sleep(0.1)
                except ValueError as e:
                    print(f"[ERROR] Invalid timestamp format: {e}")
                    return
                break

        if not found:
            print("[PAYMENT] Plate not found or already paid.")
            return

        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(entries)

    except Exception as e:
        print(f"[ERROR] Payment processing failed: {e}")

def main():
    port = detect_arduino_port()
    if not port:
        print("[ERROR] Arduino not found")
        return

    try:
        ser = serial.Serial(port, 9600, timeout=2)
        print(f"[CONNECTED] Listening on {port}")
        time.sleep(2)
        ser.reset_input_buffer()

        while True:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode().strip()
                    if line and ',' in line:
                        print(f"[SERIAL] Received: {line}")
                        plate, balance = parse_arduino_data(line)
                        if plate and balance is not None:
                            process_payment(plate, balance, ser)
            except serial.SerialException as e:
                print(f"[ERROR] Serial communication failed: {e}")
                time.sleep(1)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("[EXIT] Program terminated")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    main()