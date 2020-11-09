import numpy as np
import time
import math
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat
from pylatex.utils import italic
import subprocess
import os


def generateLatex(phrases, key, result, flag):

    geometry_options = {"tmargin": "1cm", "lmargin": "3cm", "rmargin": "3cm"}
    doc = Document(geometry_options=geometry_options)

    if(flag == "E"):
        with doc.create(Section('Encyption Input')):
            doc.append('Text: ' + "".join(phrases) + "\n")
            doc.append('Key: ' +key)

        with doc.create(Section("Matrix Mltiplications")):
            for phrase in phrases:
                M = createEncryptMatrix(key)
                messageMatrix =  np.array([[getCapitalAlphaMod(phrase[0]),getCapitalAlphaMod(phrase[1]),getCapitalAlphaMod(phrase[2])]]).astype("float64").T
                doc.append(Math(data=[r'1/' + str(26), Matrix(M), Matrix(messageMatrix), '=', Matrix(getModMatrix(M @ messageMatrix))]))
                doc.append("\n")
                doc.append("Encrypted chunk: " + getStringFromMatrix(getModMatrix(M @ messageMatrix)))

        with doc.create(Section('Encyption Result')):
            doc.append('Cipher: ' + result)
    elif(flag == "D"):
        image_filename = './LookupHill.png'
        
        with doc.create(Section('Introduction')):
            doc.append('In this project, I implement a 3  ×  3 Hill Cipher Machine in Python. This machine automatically generates LaTex reports to decipher user-input step by step. \n')
            doc.append('We will be deciphering: ' + "".join(phrases) + ' using the key: ' + key + '. \n')

        with doc.create(Section("Enryption Matrix")):
            with doc.create(Figure(position='h!')) as lookup_hill:
                lookup_hill.add_image(image_filename, width='120px')
                lookup_hill.add_caption('Lookup Table of Hill Cipher')
                
            doc.append('We use the Lookup Table above to create the Encryption Matrix below. \n')
            M = createEncryptMatrix(key)
            doc.append(Math(data=[Matrix(M)]))

        with doc.create(Section("Encryption Matrix Mod 26 Inverse")):
            '''iM = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            E21E31 = iM.copy()
            E21E31[(1,0)] = -1 * (augM[(1,0)]/augM[(0,0)])
            E21E31[(2,0)] = -1 * (augM[(2,0)]/augM[(0,0)])
            augM = E21E31.dot(augM)
            E32 = iM.copy()
            E32[(2,1)] = -1 * (augM[(2,1)]/augM[(1,1)])
            augM = E32.dot(augM)
            E23E13 = iM.copy()
            E23E13[(1,2)] = -1 * (augM[(1,2)]/augM[(2,2)])
            E23E13[(0,2)] = -1 * (augM[(0,2)]/augM[(2,2)])
            augM = E23E13.dot(augM)
            E12 = iM.copy()
            E12[(0,1)] = -1 * (augM[(0,1)]/augM[(1,1)])
            augM = E12.dot(augM)

            det = augM[(0,0)] * augM[(1,1)] * augM[(2,2)]
            if(det == 0 or math.isnan(det)):
                raise ValueError("Matrix Non-Invertible")
            #print("Det: " + str(det))
            if(egcd(int(round(det)), 26)[0] != 1):
                raise ValueError("Key Matrix determinent not co-prime with 26")

            #print("Mod inv of det: " + str(modinv(det, 26)))
            D = iM.copy()
            D[(0,0)] = 1/augM[(0,0)]
            D[(1,1)] = 1/augM[(1,1)]
            D[(2,2)] = 1/augM[(2,2)]
            augM = D.dot(augM)

            #Here are the additional steps needed to find the modular inverse of a matrix
            augM = augM * det
            augM = augM * modinv(int(round(det)), 26)

            modAugM = getModMatrix(augM[0:, 3:])

            return modAugM'''
            invMat = gaussianInverseMod26(createEncryptMatrixAug(key))
            doc.append(Math(data=[Matrix(invMat)]))
            doc.append("\n")
        
        with doc.create(Section("Matrix Multiplications")):
            for phrase in phrases:
                M = invMat
                #print(M)
                messageMatrix =  np.array([[getCapitalAlphaMod(phrase[0]),getCapitalAlphaMod(phrase[1]),getCapitalAlphaMod(phrase[2])]]).astype("float64").T
                doc.append(Math(data=[Matrix(M), Matrix(messageMatrix), '=', Matrix(getModMatrix(M @ messageMatrix))]))
                doc.append("\n")
                doc.append("Decrypted chunk: " + getStringFromMatrix(getModMatrix(M @ messageMatrix)))

        with doc.create(Section('Decryption Result')):
            doc.append('Plaintext: ' + result)

    doc.generate_pdf('full', clean_tex=False)

    subprocess.call(['open', 'full.pdf'])

def runHill(lOut = False):
    np.printoptions(precision=3, suppress=True)
    # 3-symbol hill cipher
    # key must be length 9
    # message must be multiples of 3

    choice = input("(E)ncrypt or (D)ecrypt?\n").upper()
    if choice == "E":
        text = input("Enter plaintext to encrypt\n")
        key = input("Enter key to use (must be 9 characters long)\n").upper()
        if(len(key) != 9):
            raise ValueError("Invalid key")
        for i in range((3 - (len(text) % 3)) % 3):
            text += "A"
        threePhrases = getThreeLenPhrases(text)

        cipher = ""
        for phrase in threePhrases:
            cipher += encryptMessage(phrase, key)
        print("Encrypted text is: " + cipher)
       
        if(lOut==True):
            generateLatex(threePhrases, key, cipher, "E")

    elif choice == "D":
        cipher = input("Enter encrypted text to decrypt\n")
        key = input("Enter key to use (must be 9 characters long)\n").upper()
        if(len(key) != 9):
            raise ValueError("Invalid key")
        threePhrases = getThreeLenPhrases(cipher)
        text = ""
        for phrase in threePhrases:
            text += decryptCipher(phrase, key)
        print("Decrypted text is: " + text)
        if(lOut==True):
            generateLatex(threePhrases, key, text, "D")
    else:
        print("Invalid input\n")
    

def gaussianInverseMod26(augM):
    iM = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    E21E31 = iM.copy()
    E21E31[(1,0)] = -1 * (augM[(1,0)]/augM[(0,0)])
    E21E31[(2,0)] = -1 * (augM[(2,0)]/augM[(0,0)])
    augM = E21E31.dot(augM)
    E32 = iM.copy()
    E32[(2,1)] = -1 * (augM[(2,1)]/augM[(1,1)])
    augM = E32.dot(augM)
    E23E13 = iM.copy()
    E23E13[(1,2)] = -1 * (augM[(1,2)]/augM[(2,2)])
    E23E13[(0,2)] = -1 * (augM[(0,2)]/augM[(2,2)])
    augM = E23E13.dot(augM)
    E12 = iM.copy()
    E12[(0,1)] = -1 * (augM[(0,1)]/augM[(1,1)])
    augM = E12.dot(augM)
    
    det = augM[(0,0)] * augM[(1,1)] * augM[(2,2)]
    if(det == 0 or math.isnan(det)):
        raise ValueError("Matrix Non-Invertible")
    #print("Det: " + str(det))
    if(egcd(int(round(det)), 26)[0] != 1):
        raise ValueError("Key Matrix determinent not co-prime with 26")
    
    #print("Mod inv of det: " + str(modinv(det, 26)))
    D = iM.copy()
    D[(0,0)] = 1/augM[(0,0)]
    D[(1,1)] = 1/augM[(1,1)]
    D[(2,2)] = 1/augM[(2,2)]
    augM = D.dot(augM)
    
    #Here are the additional steps needed to find the modular inverse of a matrix
    augM = augM * det
    augM = augM * modinv(int(round(det)), 26)

    modAugM = getModMatrix(augM[0:, 3:])

    return modAugM

def getCapitalAlphaMod(letter):
    return ord(letter.upper()) % 65 
    
def getAlphaFromNum(num):
    return chr(65 + num)
    
def getModMatrix(m):
    m = m.round()
    m = m % 26
    return m

def getStringFromMatrix(m):
    s = ""
    for i in range(m.shape[0]):
        #print(m[i])
        s += getAlphaFromNum(int(m[i]))
    return s
    
def createEncryptMatrix(key):
    return np.array([[getCapitalAlphaMod(key[0]),getCapitalAlphaMod(key[1]),getCapitalAlphaMod(key[2])],
    [getCapitalAlphaMod(key[3]),getCapitalAlphaMod(key[4]),getCapitalAlphaMod(key[5])],
    [getCapitalAlphaMod(key[6]),getCapitalAlphaMod(key[7]),getCapitalAlphaMod(key[8])]]).astype("float64")

def createEncryptMatrixAug(key):
    return np.array([[getCapitalAlphaMod(key[0]),getCapitalAlphaMod(key[1]),getCapitalAlphaMod(key[2]), 1, 0, 0],
    [getCapitalAlphaMod(key[3]),getCapitalAlphaMod(key[4]),getCapitalAlphaMod(key[5]), 0 ,1 ,0],
    [getCapitalAlphaMod(key[6]),getCapitalAlphaMod(key[7]),getCapitalAlphaMod(key[8]), 0, 0, 1]]).astype("float64")

def encryptMessage(message, key):
    if len(message) != 3:
        raise ValueError("Invalid message size")
        
    messageMatrix = np.array([getCapitalAlphaMod(message[0]),getCapitalAlphaMod(message[1]),getCapitalAlphaMod(message[2])]).astype("float64")
    
    keyMatrix = createEncryptMatrix(key)
    
    keyMatrixAug = createEncryptMatrixAug(key)
    
    gaussianInverseMod26(keyMatrixAug) #To check whether inverse exists/key matrix is valid. Raises ValueError if not
    
    encryptedMatrix = getModMatrix(keyMatrix.dot(messageMatrix))
    
    return getStringFromMatrix(encryptedMatrix)
    
def decryptCipher(cipher, key):
    if len(cipher) != 3:
        raise ValueError("Invalid message size")
        
    cipherMatrix = np.array([getCapitalAlphaMod(cipher[0]),getCapitalAlphaMod(cipher[1])
    ,getCapitalAlphaMod(cipher[2])]).astype("float64")
    
    keyMatrixAug = createEncryptMatrixAug(key)
    
    #Find mod 26 inverse of KeyMatrix (use Gauss Jordan + mod methods)
    modInvMat = gaussianInverseMod26(keyMatrixAug)
    #print(modInvMat)
    #Multiply that by cipherMatrix to get decryptedMatrix
    decryptedMatrix = getModMatrix(modInvMat.dot(cipherMatrix))
    #print(decryptedMatrix)
    return getStringFromMatrix(decryptedMatrix)

def getThreeLenPhrases(text):
    if(len(text) % 3 != 0):
        raise ValueError("Text not multiple of 3")

    threePhrArr = [text[::3], text[1::3], text[2::3]]
    threePhrases = []
    
    for i in range(len(threePhrArr[0])) :
        phrase = ""
        for j in range(len(threePhrArr)) :
            phrase += threePhrArr[j][i]
        threePhrases += [phrase]
    return threePhrases 

#Helper methods borrowed to calculate modular inverse of a number (https://stackoverflow.com/questions/4798654/modular-multiplicative-inverse-function-in-python)
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
    
class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

if __name__ == '__main__':
    runHill(True)
    #generateLatxex()