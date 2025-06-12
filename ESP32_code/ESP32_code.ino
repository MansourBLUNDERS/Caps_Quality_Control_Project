#include <AccelStepper.h>
#include <ESP32Servo.h>

#define STEPPER_PIN_1 16
#define STEPPER_PIN_2 17
#define STEPPER_PIN_3 18
#define STEPPER_PIN_4 19
#define SERVO_PIN 2

// Drive object for the conveyor belt
AccelStepper conveyor(AccelStepper::FULL4WIRE,
                      STEPPER_PIN_1, STEPPER_PIN_2,
                      STEPPER_PIN_3, STEPPER_PIN_4);
// Servo controller for the ejector gate
Servo gateServo;

enum EjectorState { CLOSED_READY, OPENING, HOLDING, CLOSING };
EjectorState currentState = CLOSED_READY;
unsigned long state_enter_time = 0;

// Servo movement parameters
int servo_open_degree = 70;
int servo_closed_degree = 0;
unsigned long servo_time_to_open_ms = 300;
unsigned long servo_time_to_close_ms = 1000;
unsigned long servo_hold_duration_ms = 3500;

// --------------------
// Arduino Setup
// --------------------
// Initializes serial port, servo, and stepper speed.
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32 Ready. ACK Enabled.");
  gateServo.attach(SERVO_PIN);
  gateServo.write(servo_closed_degree);
  Serial.println("STATUS: Ejector Ready");
  conveyor.setMaxSpeed(1000);
}

// --------------------
// Main Loop
// --------------------
// Continuously runs conveyor, updates ejector state,
// and processes incoming serial commands.
void loop() {
  conveyor.runSpeed();
  handleEjectorState();
  handleSerialCommands();
}

// --------------------
// Ejector State Machine
// --------------------
// Manages the transitions between OPENING, HOLDING, and CLOSING states.
void handleEjectorState() {
  if (currentState == CLOSED_READY) return;
  unsigned long elapsed = millis() - state_enter_time;
  switch (currentState) {
    case OPENING:
      if (elapsed >= servo_time_to_open_ms) {
        currentState = HOLDING;
        state_enter_time = millis();
        Serial.println("STATUS: Ejecting (Holding)");
      }
      break;
    case HOLDING:
      if (elapsed >= servo_hold_duration_ms) {
        currentState = CLOSING;
        state_enter_time = millis();
        Serial.println("STATUS: Closing Ejector...");
      }
      break;
    case CLOSING:
      if (elapsed < servo_time_to_close_ms) {
        float progress = (float)elapsed / servo_time_to_close_ms;
        int angle = servo_open_degree - (servo_open_degree * progress);
        gateServo.write(angle);
      } else {
        gateServo.write(servo_closed_degree);
        currentState = CLOSED_READY;
        Serial.println("STATUS: Ejector Closed, Ready for next trigger");
      }
      break;
  }
}

// --------------------
// Serial Command Handler
// --------------------
// Parses incoming commands (e.g., speed, trigger, config updates)
// and executes corresponding actions.
void handleSerialCommands() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) return;
    char c = cmd.charAt(0);
    int val = cmd.substring(1).toInt();
    switch (c) {
      case 'T':
        Serial.println("ACK: T received");
        if (currentState == CLOSED_READY) {
          currentState = OPENING;
          state_enter_time = millis();
          gateServo.write(servo_open_degree);
          Serial.println("STATUS: Opening Ejector...");
        }
        break;
      case 'X':
        Serial.println("ACK: X received");
        conveyor.setSpeed(0);
        break;
      case 'V':
        Serial.print("ACK: V received, Speed=");
        Serial.println(val);
        conveyor.setSpeed(val);
        break;
      case 'A':
        servo_open_degree = val;
        Serial.print("ACK: A received, servo_open_degree=");
        Serial.println(val);
        break;
      case 'O':
        servo_time_to_open_ms = val;
        Serial.print("ACK: O received, servo_time_to_open_ms=");
        Serial.println(val);
        break;
      case 'C':
        servo_time_to_close_ms = val;
        Serial.print("ACK: C received, servo_time_to_close_ms=");
        Serial.println(val);
        break;
      case 'H':
        servo_hold_duration_ms = val;
        Serial.print("ACK: H received, servo_hold_duration_ms=");
        Serial.println(val);
        break;
    }
  }
}
