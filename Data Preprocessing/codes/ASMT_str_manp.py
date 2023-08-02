####### ASSIGNMENT DATAPREPROESSING #######
## String manipulation ##
string = "Grow Gratitude"
# Q1-a Accessing Letter G of Growth
ltr_G = string[0]

# Q1-b Length of string
lgt_str = len(string) # length of string including spaces

#Q1-c Counting of G
cnt_G = string.count('G')

# Q-2
string_1 = "Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else." 
cnt_str1 = len(string_1) # total no. of characters in the string including spaces

# Q-3
string_2  = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"
# Q3-a
print(string_2[0]) # one character of word - here index 0 will be output
# Q3-b
print(string_2[0:3]) # give the 1st three character i.e. index = 0,1,2
# Q3-c
print(string_2[-3:]) # give the last 3 character i.e index = -3,-2,-1 
print(string_2[len(string_2)-3:]) # alternative way

# Q-4
string_3 = "stay positive and optimistic"
splt_str = string_3.split(' ') # split on white spaces in a list format
splt_str
# Q4-a
string_3.startswith("H")  # output is boolean 
string_3.endswith("d")
string_3.endswith("c")

# Q-5 Print statement multiple times
for i in range (108):
    print(" 🪐 " )
print(" 🪐 "  * 108) # alternative method

# Q-7 Replace Grow with Growth of
string_4 = "Grow Gratitude"
string_4.replace('Grow', 'Growth of')

# Q-8  Reverse the whole string
string_5 = ".elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A"
print(string_5[::-1]) # Print in reverse order
# Alternative
print(string_5[len(string_5)::-1]) # print starts with last index(617) and ends at 1st index(0)
# Alternative using built in function
print (''.join(reversed(string_5)))
