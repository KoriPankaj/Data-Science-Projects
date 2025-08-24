#!/usr/bin/env python
# coding: utf-8

# # DATA TYPES 

# In[2]:


# Integer
print(1)

a =15
type(a)
print(2+4)
print(12*3)


# In[3]:


# Float
print(11.5)
x = 55.45
print(x)
type(x)


# In[4]:


# Complex number
print(2+3j)
type(2+3j)


# In[5]:


# Boolean
type(True), type(False)


# In[7]:


# String Type
name = 'Pankaj'
print(name)
type (name)


# # Variables

# In[8]:


# Legal variables names
myvar = "John"
my_var = "John"
myVar = "John"
MYVAR = "John"
print(myvar)
print(my_var)
print(myVar)
print(MYVAR)


# In[11]:


# Illegal variables names
2myvar = "John"
my-var = "John"
print(2myvar)
print(my-var)


# In[12]:


# Many values to Multiple Variables
x , y , z = "orange", "banana" , "Apple"
print(x)
print(y)
print(z)


# In[13]:


# One values to Multiple variables
x = y = z = "Orange"
print(x)
print(y)
print(z)


# # Keywords

# In[15]:


import keyword
print(keyword.kwlist) # variables cannot be assign as a Keywords


# # Operators
# ## Arithmetic operators

# In[16]:


# Assign
a = 10
b = 20
a, b


# In[20]:


# addition
a+b


# In[18]:


# Substraction
a-b


# In[19]:


# Multiplication
a*b


# In[21]:


# Divison
a/b


# In[22]:


# Floor division
a//b # it rounds off the results to nearest integers


# In[23]:


# Remainders
b%a


# In[24]:


# exponential
a**b


# # Comparison Operator

# In[25]:


# It gives a bool values
a == b , a != b, a>b


# In[26]:


a <b , a>=b, a<=b


# # Assignment Operator

# In[27]:


c = a + b
print(c)
c+= b
c


# c-=b
# c

# In[30]:


c*=b
c


# In[32]:


c/=b
c


# In[34]:


c//=b
c


# # Bitwise Operators

# In[35]:


a = 60
b = 13
format(60,"b")


# # Logical operators

# In[ ]:





# # Membership operators

# In[42]:


"P" in "Python" 


# # Identity operators

# In[41]:


"y" is "python", 1 is 1,  2 is 1


# In[43]:


1 is not 1 , "hi hello" is not "hello hi"


# # DATA STRUCTURES

# ## Arrays & Vectors

# In[44]:


# Creating an array
from array import * # array is crested in pythion by importing 


# In[45]:


array1 = array('i',[10,20,30,40,50])
for x in array1:
    print(x)


# In[52]:


# insert operation
array1 = array('i', [10,20,30,40,50])
array2 = array1.insert(5,60) # insert at specidified location
for x in array1:
    print(x)


# In[48]:


# insert operation
array1 = array('i', [10,20,30,40,50])
array1.insert(5,60) # insert at specidified location
for x in array1:
    print(x)


# In[47]:


# Deletion operstion


# In[ ]:


# update operation
from array import *
array1 = array("i",[10,20,30,40,50])
array1.

