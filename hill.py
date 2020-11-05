import numpy as np

def main():
	# 3-symbol hill cipher
	# key must be length 9
	# message must be multiples of 3
	
	key = "CIPHERING"
	m = encryptMessage("TES", key)
	print(m)
	#print(decryptCipher(m, key))
	
def getCapitalAlphaMod(letter):
	return ord(letter.upper()) % 65 
	
def getAlphaFromNum(num):
	return 65 + num
	
def getModMatrix(m):
	for i in range(m.shape[0]):
		m[i] = m[i] % 26
	return m

def getStringFromMatrix(m):
	s = ""
	for i in range(m.shape[0]):
		s += chr(getAlphaFromNum(int(m[i])))
	return s
	
def encryptMessage(message, key):
	if len(message) != 3:
		raise ValueError("Invalid message size")
		
	messageMatrix = np.array([getCapitalAlphaMod(message[0]),getCapitalAlphaMod(message[1])
	,getCapitalAlphaMod(message[2])])
	
	keyMatrix = np.array([[getCapitalAlphaMod(key[0]),getCapitalAlphaMod(key[1]),getCapitalAlphaMod(key[2])],
	[getCapitalAlphaMod(key[3]),getCapitalAlphaMod(key[4]),getCapitalAlphaMod(key[5])],
	[getCapitalAlphaMod(key[6]),getCapitalAlphaMod(key[7]),getCapitalAlphaMod(key[8])]])
	
	encryptedMatrix = getModMatrix(keyMatrix.dot(messageMatrix))
	
	return getStringFromMatrix(encryptedMatrix)
	
def decryptCipher(cipher, key):
	if len(cipher) != 3:
		raise ValueError("Invalid message size")
		
	cipherMatrix = np.array([getCapitalAlphaMod(cipher[0]),getCapitalAlphaMod(cipher[1])
	,getCapitalAlphaMod(cipher[2])])
	
	keyMatrix = np.array([[getCapitalAlphaMod(key[0]),getCapitalAlphaMod(key[1]),getCapitalAlphaMod(key[2])],
	[getCapitalAlphaMod(key[3]),getCapitalAlphaMod(key[4]),getCapitalAlphaMod(key[5])],
	[getCapitalAlphaMod(key[6]),getCapitalAlphaMod(key[7]),getCapitalAlphaMod(key[8])]])
	
	#Find mod 26 inverse of KeyMatrix (use Gauss Jordan + mod methods)
	#Multiply that by cipherMatrix to get decryptedMatrix

	#return getStringFromMatrix(decryptedMatrix)
	
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

if __name__ == '__main__':
	main()
