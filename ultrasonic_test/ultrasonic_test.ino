//www.elegoo.com
//2016.09.23

#define ENA 5
#define ENB 11
#define IN1 6
#define IN2 7
#define IN3 8
#define IN4 9

#define ECHO A4
#define TRIG A5
#define SENSOR_OUT 3

class Motor {
private:
  int enable;
  int in1;
  int in2;

public:
  Motor(int enable, int in1, int in2) {
    this->enable = enable;
    this->in1 = in1;
    this->in2 = in2;

    pinMode(enable, OUTPUT);
    pinMode(in1, OUTPUT);
    pinMode(in2, OUTPUT);

    analogWrite(enable, 0);
    digitalWrite(in1, 0);
    digitalWrite(in2, 0);
  }

  void setSpeed(int value) {
    if (value == 0) {  // stop
      analogWrite(this->enable, 0);
      digitalWrite(this->in1, 0);
      digitalWrite(this->in2, 0);

    } else if (value > 0) {  // foward
      analogWrite(this->enable, value);
      digitalWrite(this->in1, 1);
      digitalWrite(this->in2, 0);

    } else {  // backward
      analogWrite(this->enable, -value);
      digitalWrite(this->in1, 0);
      digitalWrite(this->in2, 1);
    }
  }
};

class Ultrasonic {
private:
  int echo;
  int trig;
  int pin_out;
  int thresh;

public:
  Ultrasonic(int echo, int trig, int pin_out) {
    this->echo = echo;
    this->trig = trig;
    this->thresh = 100;
    this->pin_out = pin_out;

    pinMode(echo, INPUT);
    pinMode(trig, OUTPUT);
    pinMode(pin_out, OUTPUT);

    digitalWrite(this->pin_out, 0);
  }

  int getDistance() {
    digitalWrite(this->trig, LOW);
    delayMicroseconds(2);
    digitalWrite(this->trig, HIGH);
    delayMicroseconds(20);
    digitalWrite(this->trig, LOW);

    return (int)(pulseIn(this->echo, HIGH) / 58);
  }

  void setThreshold(int thresh) {
    this->thresh = thresh;
  }

  bool checkThreshold() {
    int dist = this->getDistance();

    if (dist > 350) {
      return;
    }
    
    if (dist < this->thresh) {
      digitalWrite(this->pin_out, HIGH);
      return true;
    } else {
      digitalWrite(this->pin_out, LOW);
      return false;
    }
  }
};


/*define logic control output pin*/
Motor rightSide = Motor(ENB, IN3, IN4);
Motor leftSide = Motor(ENA, IN2, IN1);

Ultrasonic sensor = Ultrasonic(ECHO, TRIG, SENSOR_OUT);

int right_val = 0;
int left_val = 0;

int ultra_time = 0;
bool obstacle = false;

byte data[5];

/*put your setup code here, to run once*/
void setup() {
  Serial.begin(9600);  //Open the serial port and set the baud rate to 9600

  rightSide.setSpeed(0);
  leftSide.setSpeed(0);
}

/*put your main code here, to run repeatedly*/
void loop() {
  /*
    first nibble: 
      0x1 indicates that byte[1] will be a new ultrasonic threshold
    second nibble:
      ignored if the first nibble != 0x0
      indicates the sign of byte[1] and byte[2]
        0x1 indicating byte[1] is negative 
        0x2 indicating byte[2] is negative 
        0x3 indicating both are negative 
  */
  if (Serial.available() > 0) {
    Serial.readBytesUntil('\n', data, 5);

    // motor values
    right_val = data[1];
    left_val = data[2];

    if ((data[0] & 0x1) != 0) {
      right_val *= -1;
    }

    if ((data[0] & 0x2) != 0) {
      left_val *= -1;
    }
    
  }

  
    if (sensor.checkThreshold()) {
      rightSide.setSpeed(0);
      leftSide.setSpeed(0);
      obstacle = true;
    } else {
      obstacle = false;
    }

  

  if (!obstacle) {
    rightSide.setSpeed(right_val);
    leftSide.setSpeed(left_val);
  }

  delay(200);
}
