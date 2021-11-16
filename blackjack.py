import random
import time
with open('E:\python_files\slackjack.txt','r') as cash:
    cash=int(cash.read())

    data={'ace':11,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':10,'Q':10,'K':10}
    print('cash=',cash)
    input('welcome to blackjack, press Enter...(dealer must hit on 16)')
    y=True
    while y:
        cash=int(cash)
        if cash<=0:

            print('you run out of cash!we give u 20 extra cash...')
            cash = 20
        print('bet?max=',cash)
        w=int(input())
        if w>cash:
            print("u don't have enough money:|")
            break

        dealer=random.choice(list(data))
        player=random.choice(list(data))


        print(f'dealer : {dealer}\n you: {player}')
        value_dealer=data[dealer]
        value_player=data[player]


        x=True

        while x:

            q=input('take ? :y/n ')


            if q=='y':

             player = random.choice(list(data))

             print(f'dealer : {dealer}\n you: {player}')
             value_player=value_player+data[player]
             print(f'sum :{value_player}')




            if value_player==21:
                p=1
                print('player wins\n________')
                break
            if value_player>21:
                  p=0
                  print('dealer wins\n________')
                  break
            if q=='n':
                break
        if q=='n':
         while value_dealer<17 :
            dealer = random.choice(list(data))
            value_dealer = value_dealer + data[dealer]
            print(f'dealer : {dealer}\n{value_dealer}\n you: {value_player}')
            time.sleep(1)
            if value_dealer >value_player and value_dealer<21:
                p=0
                print('dealer wins\n________')
                break
            if value_dealer>21:
                p=1
                print('player wins\n________')

            if value_dealer==21:
                p=0
                print('dealer wins\n________')
        if p==0:
            cash=cash-w
        if p==1:
            cash=cash+w

        with open('E:\python_files\slackjack.txt','w') as cashz:
         cash=str(cash)
         cashz.write(cash)
