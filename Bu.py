#!/usr/bin/python3
# coding: utf-8

"""
    This module used to transliterate Unicode Arabic to Buckwalter
"""

toBuckwalter = [0]*(7*256)
toBuckwalter[ 32 ] =  32 	#     
toBuckwalter[ 9 ] =  9 	#  	 	
toBuckwalter[ 10 ] =  10 	#  
 

toBuckwalter[ 13 ] =  13 	#   
toBuckwalter[ 63 ] =  63 	#  ? ?
toBuckwalter[ 46 ] =  46 	#  . .
toBuckwalter[ 44 ] =  44 	#  , ,
toBuckwalter[ 44 ] =  44 	#  , ,
toBuckwalter[ 58 ] =  58 	#  : :
toBuckwalter[ 40 ] =  40 	#  ( (
toBuckwalter[ 41 ] =  41 	#  ) )
toBuckwalter[ 33 ] =  33 	#  ! !
toBuckwalter[ 34 ] =  34 	#  " "
toBuckwalter[ 171 ] =  171 	#  « «
toBuckwalter[ 187 ] =  187 	#  » »
toBuckwalter[ 1567 ] =  63 	#  ؟ ?
toBuckwalter[ 1548 ] =  44 	#  ، ,
toBuckwalter[ 1563 ] =  59 	#  ؛ ;
toBuckwalter[ 1642 ] =  37 	#  ٪ %
toBuckwalter[ 1643 ] =  59 	#  ٫ ;
toBuckwalter[ 1644 ] =  44 	#  ٬ ,
toBuckwalter[ 48 ] =  48 	#  0 0
toBuckwalter[ 49 ] =  49 	#  1 1
toBuckwalter[ 50 ] =  50 	#  2 2
toBuckwalter[ 51 ] =  51 	#  3 3
toBuckwalter[ 52 ] =  52 	#  4 4
toBuckwalter[ 53 ] =  53 	#  5 5
toBuckwalter[ 54 ] =  54 	#  6 6
toBuckwalter[ 55 ] =  55 	#  7 7
toBuckwalter[ 56 ] =  56 	#  8 8
toBuckwalter[ 57 ] =  57 	#  9 9
toBuckwalter[ 1632 ] =  48 	#  ٠ 0
toBuckwalter[ 1633 ] =  49 	#  ١ 1
toBuckwalter[ 1634 ] =  50 	#  ٢ 2
toBuckwalter[ 1635 ] =  51 	#  ٣ 3
toBuckwalter[ 1636 ] =  52 	#  ٤ 4
toBuckwalter[ 1637 ] =  53 	#  ٥ 5
toBuckwalter[ 1638 ] =  54 	#  ٦ 6
toBuckwalter[ 1639 ] =  55 	#  ٧ 7
toBuckwalter[ 1640 ] =  56 	#  ٨ 8
toBuckwalter[ 1641 ] =  57 	#  ٩ 9
toBuckwalter[ 1776 ] =  48 	#  ۰ 0
toBuckwalter[ 1777 ] =  49 	#  ۱ 1
toBuckwalter[ 1778 ] =  50 	#  ۲ 2
toBuckwalter[ 1779 ] =  51 	#  ۳ 3
toBuckwalter[ 1780 ] =  52 	#  ۴ 4
toBuckwalter[ 1781 ] =  53 	#  ۵ 5
toBuckwalter[ 1782 ] =  54 	#  ۶ 6
toBuckwalter[ 1783 ] =  55 	#  ۷ 7
toBuckwalter[ 1784 ] =  56 	#  ۸ 8
toBuckwalter[ 1785 ] =  57 	#  ۹ 9
toBuckwalter[ 1569 ] =  39 	#  ء '
toBuckwalter[ 1570 ] =  124 	#  آ |
toBuckwalter[ 1571 ] =  62 	#  أ >
toBuckwalter[ 1572 ] =  38 	#  ؤ &
toBuckwalter[ 1573 ] =  60 	#  إ <
toBuckwalter[ 1574 ] =  125 	#  ئ }
toBuckwalter[ 1575 ] =  65 	#  ا A
toBuckwalter[ 1576 ] =  98 	#  ب b
toBuckwalter[ 1577 ] =  112 	#  ة p
toBuckwalter[ 1578 ] =  116 	#  ت t
toBuckwalter[ 1579 ] =  118 	#  ث v
toBuckwalter[ 1580 ] =  106 	#  ج j
toBuckwalter[ 1581 ] =  72 	#  ح H
toBuckwalter[ 1582 ] =  120 	#  خ x
toBuckwalter[ 1583 ] =  100 	#  د d
toBuckwalter[ 1584 ] =  42 	#  ذ *
toBuckwalter[ 1585 ] =  114 	#  ر r
toBuckwalter[ 1586 ] =  122 	#  ز z
toBuckwalter[ 1587 ] =  115 	#  س s
toBuckwalter[ 1588 ] =  36 	#  ش $
toBuckwalter[ 1589 ] =  83 	#  ص S
toBuckwalter[ 1590 ] =  68 	#  ض D
toBuckwalter[ 1591 ] =  84 	#  ط T
toBuckwalter[ 1592 ] =  90 	#  ظ Z
toBuckwalter[ 1593 ] =  69 	#  ع E
toBuckwalter[ 1594 ] =  103 	#  غ g
toBuckwalter[ 1600 ] =  95 	#  ـ _
toBuckwalter[ 1601 ] =  102 	#  ف f
toBuckwalter[ 1602 ] =  113 	#  ق q
toBuckwalter[ 1603 ] =  107 	#  ك k
toBuckwalter[ 1604 ] =  108 	#  ل l
toBuckwalter[ 1605 ] =  109 	#  م m
toBuckwalter[ 1606 ] =  110 	#  ن n
toBuckwalter[ 1607 ] =  104 	#  ه h
toBuckwalter[ 1608 ] =  119 	#  و w
toBuckwalter[ 1609 ] =  89 	#  ى Y
toBuckwalter[ 1610 ] =  121 	#  ي y
toBuckwalter[ 1611 ] =  70 	#  ً F
toBuckwalter[ 1612 ] =  78 	#  ٌ N
toBuckwalter[ 1613 ] =  75 	#  ٍ K
toBuckwalter[ 1614 ] =  97 	#  َ a
toBuckwalter[ 1615 ] =  117 	#  ُ u
toBuckwalter[ 1616 ] =  105 	#  ِ i
toBuckwalter[ 1617 ] =  126 	#  ّ ~
toBuckwalter[ 1618 ] =  111 	#  ْ o
toBuckwalter[ 1648 ] =  96 	#  ٰ `
toBuckwalter[ 1649 ] =  123 	#  ٱ {
toBuckwalter[ 1670 ] =  74 	#  چ J

toUnicode = dict()
for uni,buck in enumerate(toBuckwalter):
    if buck !=0:
        if (buck in toUnicode and buck != 44 and buck != 63 and 
               buck != 59 and
               not (uni >=1632  and uni <= 1641) and
               not (uni >=1776  and uni <= 1785)   ):
            print(buck, 'has code', toUnicode[buck], 'and', uni)
            cry()
        else:
            toUnicode[chr(buck)] = chr(uni)
    toUnicode[chr(44)] = chr(1548)
    toUnicode[chr(63)] = chr(1567)

def intoBuckwalter(s):
    answer = ""
    for c in s:
        t = toBuckwalter[ord(c)]
        if t ==0:
            answer += c
        else:
            answer += chr(t)
    return answer

def fromBuckwalter(s):
    answer = u''
    for c in s:
        if c in toUnicode:
            answer += toUnicode[c]
        else:
            answer += '?'
    return answer

if __name__ == '__main__':
    print(fromBuckwalter('mtlk'))
    print(intoBuckwalter(u'ابن'))
