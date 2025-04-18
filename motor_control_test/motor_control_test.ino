//www.elegoo.com
//2016.09.23

#define ENA 5
#define ENB 11
#define IN1 6
#define IN2 7
#define IN3 8
#define IN4 9

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




/*define logic control output pin*/
Motor rightSide = Motor(ENB, IN3, IN4);
Motor leftSide = Motor(ENA, IN2, IN1);

int Echo = A4;
int Trig = A5;
int ABS = 150;

int dist = 0;

/*define forward function*/
void _mForward() {
  rightSide.setSpeed(100);
  leftSide.setSpeed(100);
  Serial.println("Forward");
}
/*define back function*/
void _mBack() {
  rightSide.setSpeed(-100);
  leftSide.setSpeed(-100);
  Serial.println("Back");
}

/*define left function*/
void _mleft() {
  rightSide.setSpeed(0);
  leftSide.setSpeed(100);
  Serial.println("Left");
}

/*define right function*/
void _mright() {
  //digitalWrite(ENA,HIGH);
  rightSide.setSpeed(100);
  leftSide.setSpeed(0);
  Serial.println("Right");
}

/*Ultrasonic distance measurement Sub function*/
int Distance_test() {
  digitalWrite(Trig, LOW);
  delayMicroseconds(2);
  digitalWrite(Trig, HIGH);
  delayMicroseconds(20);
  digitalWrite(Trig, LOW);
  float Fdistance = pulseIn(Echo, HIGH);
  Fdistance = Fdistance / 58;
  return (int)Fdistance;
}

/*put your setup code here, to run once*/
void setup() {
  Serial.begin(9600);  //Open the serial port and set the baud rate to 9600
                       /*Set the defined pins to the output*/
  pinMode(Echo, INPUT);
  pinMode(Trig, OUTPUT);

  rightSide.setSpeed(0);
  leftSide.setSpeed(0);
}

/*put your main code here, to run repeatedly*/
void loop() {
  _mForward();
  delay(1000);
  rightSide.setSpeed(0);
  leftSide.setSpeed(0);
  delay(1000);
  rightSide.setSpeed(0);
  leftSide.setSpeed(0);
  delay(1000);
  _mBack();
  delay(1000);
  rightSide.setSpeed(0);
  leftSide.setSpeed(0);
  delay(1000);
  _mleft();
  delay(1000);
  rightSide.setSpeed(0);
  leftSide.setSpeed(0);
  delay(1000);
  _mright();
  delay(1000);
  rightSide.setSpeed(0);
  leftSide.setSpeed(0);
  delay(1000);

  dist = Distance_test();
  Serial.println(dist);

  delay(500);
}
