package week_01;
/** Mã Caesar: dịch chuyển A-Z/a-z; ký tự ngoài A-Z được giữ nguyên. */
public final class CaesarCipher {
private CaesarCipher() { }
public static String encrypt(String text, int key) {
return transform(text, key);
}
public static String decrypt(String text, int key) {
return transform(text, -key);
}
private static String transform(String text, int key) {
StringBuilder result = new StringBuilder();
int shift = Math.floorMod(key, 26);
for (char c : text.toCharArray()) {
char base;
if (c >= 'A' && c <= 'Z') {
base = 'A';
} else if (c >= 'a' && c <= 'z') {
base = 'a';
} else {
result.append(c);
continue;
}
int original = c - base;
int encoded = Math.floorMod(original + shift, 26);
result.append((char) (base + encoded));
}
return result.toString();
}
}