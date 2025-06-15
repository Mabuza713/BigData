

def zad2():
    for i in range (1, 31):
        if i%2==0:
            print(i)
zad2()

def zad3():
    x=12
    suma=0
    while(suma!=x):
        suma+=1
    print("Done 3")
zad3()

def zad4():
    x=12
    suma=0
    for i in range(1, x):
        if i%2==0:
            suma+=i
    print("Done 4")
zad4()

def zad5():
    for i in range (100, -101, -1):
        if i%2==0 and i%3!=0 and i%8!=0:
            print(i)
zad5()

def zad6():
    tablica = []
    for i in range(1, 6):
        wiersz = []
        for j in range(1, 6):
            wiersz.append(min(i, j))
        tablica.append(wiersz)
    print(tablica)
zad6()

def zad7(lista, n):
    wynik = [[] for _ in range(n)]
    for i, element in enumerate(lista):
        wynik[i % n].append(element)
    print (wynik)

dane = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
zad7(dane, 4)

def zad8(lista1, lista2):
    print(lista1[:-1] + lista2)

lista1 = [100, 90, 80, 70, 60, 50]
lista2 = [49, 39, 29, 19]
zad8(lista1, lista2)

def zad9(dane, s):
    print ([f"{s} {i}" for i in dane])
dane = ['A', 'B', 'C', 'D']
string = 'Exit'
zad9(dane, string)

def zad10(lista):
    print ([i[:-1] + (0,) for i in lista])

lista=[(1, 2, 3), (4, 5, 6), (7, 8, 9)]
zad10(lista)

def zad11(lista):
    pom=[]
    for i in lista:
        if i != ():
            pom.append(i)
    print(pom)

dane11= [(), (), ('',), ('i1', 'i2'), ('i1', 'i2', 'i3'), ('i4')]
zad11(dane11)

def zad12(dict):
    iloczyn=1
    for i in dict:
        iloczyn*=dict[i]
    print (iloczyn)
dane12={ 'f1': 4.8, 'f2': 2.4, 'f3': 1.2, 'f4': 0.6}
zad12(dane12)

def zad13(n):
    dict={i: i**4 for i in range(1, n+1)}
    print (dict)

n=12
zad13(n)

def zad14(dict):
    print(set(dict.values()))

dane14={1: 'A201', 2: 'B218', 3:'H018', 4:'B218', 5:'H018', 6: 'G123', 7: 'A007', 8: 'G230'}
zad14(dane14)  

