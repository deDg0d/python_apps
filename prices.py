import requests
from bs4 import BeautifulSoup

url=requests.get('https://www.tgju.org/').text
soup= BeautifulSoup(url,'lxml')
#dollar section
dollar=soup.find('li',{'id':'l-price_dollar_rl'})

dollar=dollar.find('span',{'class':'info-price'}).text

dollar=str(dollar)



dollar=dollar.replace(',','')
dollar=int(dollar)
dollar=dollar/10



#oz gold section va gold_in_toman va onegrm gold be toman
gold=soup.find('li',{'id':"l-ons"}).find('span',{'class':'info-price'}).text
gold=str(gold)

gold=gold.replace(',','')
gold=gold.replace('.','')
gold=int(gold)
gold=gold/100
onegrmgold=gold/28
onegrmgold=onegrmgold*dollar
gold_in_toman=gold*dollar



#gold_coin rial
goldcoin=soup.find('li',{'id':"l-irec_future"}).find('span',{'class':'info-price'}).text

goldcoin=str(goldcoin)




#gold in dollar va gold in toman  dollar and gold full coin
#currencies rates and names variables


urs=requests.get('https://www.x-rates.com/table/?from=USD&amount=1').text
cur=BeautifulSoup(urs,'lxml')

cur=cur.find('table',{'class':'ratesTable'})

def name_rates ():

    for y in cur.find_all('tr'):
      names=y.td
      names=str(names)
      names=names.replace('<td>','')
      names = names.replace("</td>", "")
      rates=y.find('a')
      rates=str(rates)
      rates=rates[-14:]
      rates=rates.replace('">','')
      rates=rates.replace('</a>','')
      rates = rates.replace('>', '')


      rates=rates.replace('.','')
      #print(rates)

      try:
         rates=int(rates)


         rates=rates/1000000
         rates=dollar/rates

         print(names +' : ')
         print(rates)
      except:
          None
x=input('please enter the value:\n1:dollar today: \n2:gold today: \n3:gold in toman: \n4:1grm gold:  \n5:gold coin: \n6:country currencies:  ')
x=int(x)
if x==1:

   print(dollar)

elif x==2:
   print(gold)
elif x==3:
    print(gold_in_toman)
elif x==4:
    print(onegrmgold)
elif x==5:
    print(goldcoin)
elif x==6:
    print(name_rates())

