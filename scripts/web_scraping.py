#!/usr/bin/env python
# coding: utf-8

# # Web scraping

# In[3]:


# Important packages
import chromedriver_binary
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import io

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')


# In[2]:


# Let's use ChromeDriverManager to install 
# the correct version of chromedriver
driver = webdriver.Chrome(ChromeDriverManager(version = '87.0.4280.88').install())

# acess the researchgate's login webpage
driver.get('https://www.researchgate.net/login')


# In[4]:


# Locate the login tab and insert the account's email
username = driver.find_element_by_name("login")
username.clear()
username.send_keys("2001110583@stu.pku.edu.cn")
time.sleep(1)

# Locate the password tab and insert the password
password = driver.find_element_by_name("password")
password.clear()
password.send_keys("dcdd21de")
time.sleep(1)

# Click on the 'Log in' button
driver.find_element_by_class_name("nova-c-button__label").click()


# In[5]:


# Clilck in 'more'
more = driver.find_element_by_class_name("nova-c-button__label").click()
time.sleep(1)
# click in 'Search'
search = driver.find_element_by_link_text("Search").click()


# In[6]:


# search (and click on) the 'Publication' button
if driver.find_element_by_xpath("//*[contains(text(), 'Publications')]").is_displayed():
    time.sleep(1)
    driver.find_element_by_xpath("//*[contains(text(),'Publications')]").click()


# In[7]:


### search for specific words in the publications 
# search for the search bar
search_bar = driver.find_element_by_css_selector("input[placeholder='Search ResearchGate']")
time.sleep(1)

# clear whichever content it has
search_bar.clear()
time.sleep(1)

# search for "taenite" in the search bar
search_bar.send_keys("taenite")
time.sleep(1)

# submit the search
search_bar.send_keys(Keys.RETURN)


# In[8]:


# pubs and abstracts are the classes in the lxml file
# where publication's name and abstract (respectively) are found
# the paper's name is appended in the 'titles' list
# and the abstract in 'abstracts' 

abstracts = []
titles = []

pubs = 'nova-e-text nova-e-text--size-m nova-e-text--family-sans-serif nova-e-text--spacing-none nova-e-text--color-inherit nova-e-text--clamp-3 nova-v-publication-item__description'
abstract = 'nova-e-text nova-e-text--size-l nova-e-text--family-sans-serif nova-e-text--spacing-none nova-e-text--color-inherit nova-e-text--clamp-3 nova-v-publication-item__title'


# In[9]:


# Download ~ 500 publications related with the 
# searched word 'taenite'

title = []

while len(abstracts) < 500:    
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'lxml')
    content = soup.find_all('div', class_ = pubs)
    title = soup.find_all('div', class_ = abstract)    
    for i in content:
        if i.get_text() in abstracts:
            pass
        elif not i.get_text():
            abstracts.append("Null")
        else:
            abstracts.append(i.get_text())
            
    for j in title:
        if j.get_text() in titles:
            pass
        else:
            titles.append(j.get_text())            
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")    
    time.sleep(5)              
   


# In[10]:


# save the content as .txt files
# with the publication's name as the fle name
# (with spaces replaced by _)
for i in range(len(titles)):
    pap_title = titles[i]
    pap_title_2 = "_".join(pap_title.split())   
    try:
        with io.open('/Users/josea/Desktop/data/{}.txt'.format(pap_title_2), 'w', encoding = 'utf-8') as f:
            f.write('Title: {} \n'.format(pap_title))
            f.write('\n')
            f.write('\n')
            f.write('Abstract: {} \n'.format(names[i]))
    except:
        pass


# In[ ]:




