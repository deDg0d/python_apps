import requests
from bs4 import BeautifulSoup

headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
url=requests.get('https://777score.com/live?matchList=actual',headers=headers).text
soup=BeautifulSoup(url,'lxml')



res=soup.find_all('div',{'class':'tournaments-match'})

for res in soup.findAll('div', {'class': 'tournaments-match'}):

     #
    try:
      #league = soup.find('h3').text
      league = soup.find('a', {'class': 'tournaments-title_a'}).text
      time = res.find('span', {'class': 'matchMinute matchState matchStatus matchLive'}).text
    except:
        None
    league=soup.find('a').text

    name1=res.find('div',{'class':'team-info teamHost'}).text
    name2=res.find('div',{'class':'team-info teamGuest'}).text



    result1=res.find('span',{'class','point scoreHost'}).text
    result2=res.find('span',{'class','point scoreGuest'}).text
    print(league + name1.strip() +' '+ result1.strip() +' '+':'+' '+ result2.strip()+' '+ name2.strip() + ' ' +time.strip())

'''''
print(res)

'''''
x=input('press enter x to finish')
