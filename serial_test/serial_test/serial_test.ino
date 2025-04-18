int iter = 0;
byte data[5];

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    Serial.readBytesUntil('\n', data, 5);

    Serial.print("You sent me: ");
    Serial.print(data[0], HEX);
    Serial.print("|");
    Serial.print(data[1], HEX);
    Serial.print("|");
    Serial.print(data[2], HEX);
    Serial.print("|");

    int val1 = data[1];
    int val2 = data[2];

    if((data[0] & 0x1) == 1){
      val1 *= -1;
    }

    if((data[0] & 0x2) == 2){
      val2 *= -1;
    }

    Serial.print(val1);
    Serial.print("|");
    Serial.print(val2);
    Serial.print("[");
    Serial.print(iter++);
    Serial.println("]");
  }
}