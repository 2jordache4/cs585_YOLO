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
  int thresh;

public:
  Ultrasonic(int echo, int trig) {
    this->echo = echo;
    this->trig = trig;

    pinMode(echo, INPUT);
    pinMode(trig, OUTPUT);
  }

  int getDistance() {
    digitalWrite(this->trig, LOW);
    delayMicroseconds(2);
    digitalWrite(this->trig, HIGH);
    delayMicroseconds(20);
    digitalWrite(this->trig, LOW);

    return (int)(pulseIn(this->echo, HIGH) / 58);
  }
};


/*define logic control output pin*/
Motor rightSide = Motor(ENB, IN3, IN4);
Motor leftSide = Motor(ENA, IN2, IN1);

Ultrasonic sensor = Ultrasonic(ECHO, TRIG);

int right_val = 0;
int left_val = 0;

int ABS = 150;
int dist = 0;

byte data[5];

/*put your setup code here, to run once*/
void setup() {
  Serial.begin(9600);  //Open the serial port and set the baud rate to 9600

  rightSide.setSpeed(0);
  leftSide.setSpeed(0);
}

/*put your main code here, to run repeatedly*/
void loop() {
  if (Serial.available() > 0) {
    Serial.readBytesUntil('\n', data, 5);

    right_val = data[1];
    left_val = data[2];

    if ((data[0] & 0x1) == 1) {
      right_val *= -1;
    }

    if ((data[0] & 0x2) == 2) {
      left_val *= -1;
    }
  }

  rightSide.setSpeed(right_val);
  leftSide.setSpeed(left_val);

  delay(100);
}
