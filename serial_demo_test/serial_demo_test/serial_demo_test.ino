void setup() {
  Serial.begin(9600);
}

void loop() {
  int iter = 0;

  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    Serial.print("You sent me: ");
    Serial.print(data);
    Serial.print("[");
    Serial.print(iter++);
    Serial.println("]");
  }
}