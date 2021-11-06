import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import locale
import math
# from scipy import stats

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
df = pd.read_csv(url, sep='\t')



# print(df[lambda x: x['order_id'] > 400])
# print(result)
def task1():
    print(df.shape[0], 'Number of observations')
    return 1

def task2():
    print(df.columns.tolist())
    return 2

def task3():
    # print(stats.mode(df["item_name"]))
    # print(df['item_name'].value_counts()[:3].index.tolist())
    print(df['item_name'].value_counts()[df['item_name'].value_counts() == df['item_name'].value_counts().max()])

def task4():
    value_counts = df['item_name'].value_counts()
    df_val_counts = pd.DataFrame(value_counts)
    frequency_df = df_val_counts.reset_index()
    frequency_df.columns = ['item_name', 'counts']
    frequency_df.set_index('item_name')['counts'].plot(kind='bar')
    # frequency_df.hist()
    # plt.hist(x, density=True, bins=30)  # density=False would make counts
    # plt.ylabel('item_name')
    # plt.xlabel('counts');
    plt.show()
    print (frequency_df)

def task5():
    df['item_price']=df.item_price.map(lambda x: locale.atof(x.strip('$')))
    print (df.item_price)

def task6():
    df['item_price']=df.item_price.map(lambda x: locale.atof(x.strip('$')))
    sum = df.groupby(['item_name'], as_index=False, sort=False)['item_price'].sum().sort_values(['item_price'], ascending=False)
    sum.set_index('item_name')['item_price'].plot(kind='bar')
    plt.show()
    # print (sum)

def task7():
    df['item_price']=df.item_price.map(lambda x: locale.atof(x.strip('$')))
    average_sum_of_order = df['item_price'].sum()/df.shape[0]
    print ('$'+str(round(average_sum_of_order,2))+' average sum of an order')

    quantity_of_each_name = df['item_name'].value_counts(sort=False)
    list_of_unique_items_sum = df.groupby(['item_name'], as_index=False, sort=False)['item_price'].sum().item_price

    avg_sum = [list_of_unique_items_sum[i]/quantity_of_each_name[i] for i in range(0,len(quantity_of_each_name)-1)]
    avg_order_sum = sum(avg_sum)/len(avg_sum)
    print('$'+str(round(avg_order_sum))+' average sum of an order')

def task8():
    df['item_price']=df.item_price.map(lambda x: locale.atof(x.strip('$')))
    # avg_quantity = df.groupby(['item_name'])['quantity'].sum()
    group = df.groupby(['order_id'])['order_id'].count()

    # average_q = df['quantity'].sum()/df.shape[0]
    average_q = group.sum()/len(group)
    minimum_q = group.min()
    maximum_q = group.max()
    median_q = np.median(group)
    print('avg quantity: '+str(round(average_q, 2)),'  min quantity: '+str(minimum_q),'  max quantity: ' +str(maximum_q), '  median quantity: '+ str(median_q))

def task9():
    quantity_of_steaks = [val for val in df.item_name if 'Steak' in val]
    # quantity_of_hot_steaks = [val for val, desc in zip(df.item_name, df.choice_description) if 'Steak' in val and 'Hot' in desc and 'Roasted' in desc]
    # quantity_of_medium_steaks = [val for val, desc in zip(df.item_name, df.choice_description) if 'Steak' in val and 'Medium' in desc and 'Roasted' in desc]
    # quantity_of_mild_steaks = [val for val, desc in zip(df.item_name, df.choice_description) if 'Steak' in val and 'Mild' in desc and 'Roasted' in desc]

    def cutDescription(description):
        if ' [' in description: return description[:description.index(' [')]
        else: return description

    steaks_hot = [cutDescription(description) for val, description in zip(df.item_name, df.choice_description) if 'Steak' in val and 'Hot' in description]
    steaks_medium = [cutDescription(description) for val, description in zip(df.item_name, df.choice_description) if 'Steak' in val and 'Medium' in description]
    steaks_mild = [cutDescription(description) for val, description in zip(df.item_name, df.choice_description) if 'Steak' in val and 'Mild' in description]

    steaks_hot = pd.DataFrame(steaks_hot).value_counts()
    steaks_medium = pd.DataFrame(steaks_medium).value_counts()
    steaks_mild = pd.DataFrame(steaks_mild).value_counts()

    print(steaks_hot)
    print(steaks_medium)
    print(steaks_mild)

    print (round(len(quantity_of_steaks)/len(df)*100), '% of all orders are steaks')
    print(round(steaks_hot.sum()/len(quantity_of_steaks)*100), '% of all steaks are hot')
    print(round(steaks_medium.sum()/len(quantity_of_steaks)*100), '% of all steaks are medium')
    print(round(steaks_mild.sum()/len(quantity_of_steaks)*100), '% of all steaks are mild')

    print('Сначала выводится распределение по видам прожарок в соответствии с порциями, далее обшая частота')

    # print (round(len(quantity_of_hot_steaks)/len(quantity_of_steaks)*100), '% of all steaks are hot')
    # print (round(len(quantity_of_medium_steaks)/len(quantity_of_steaks)*100), '% of all steaks are medium')
    # print (round(len(quantity_of_mild_steaks)/len(quantity_of_steaks)*100), '% of all steaks are mild')

def task10():
    df['item_price']=df.item_price.map(lambda x: locale.atof(x.strip('$')))
    df['item_price_in_rur'] = ['₽'+str(round(val*71.15)) for val in df.item_price]
    print (df)

def task11():
    print(df['item_name'].value_counts())
    steaks_only = [val for val in df.item_name if 'Steak' in val]
    print (pd.DataFrame(steaks_only).value_counts())
    print ('Здесь учтены все прожарки, по отдельности можно посмотреть task9')

def isNaN(num):
    return num != num

def task12():

    df['item_price']=df.item_price.map(lambda x: locale.atof(x.strip('$')))
    prices_for_chips = [float(price) for val, des, price in zip(df.item_name, df.choice_description, df.item_price) if 'Chips'==val and isNaN(des)]
    prices_for_chips = list(set(prices_for_chips))
    input_price = input (f"Введите желаемую стоимость `Chips` {prices_for_chips}: \n ")

    chosen_price = 0
    for i in range(0, len(prices_for_chips)):
        if (float(input_price) == prices_for_chips[i]):
            chosen_price = float(input_price)
    if (chosen_price == 0): print('incorrect price, try pick again')
    # else: print('chosen_price:', chosen_price)

    def lowCost(price, chosen_price):
        if (price - chosen_price < 0):
            return ' - *free*'
        else: return ' - $'+str(round(price - chosen_price, 2))

    with_chips = [val[10:] + lowCost(price, chosen_price) for val, description, price in zip(df.item_name, df.choice_description, df.item_price) if isNaN(description) and 'Chips' in val and 'and' in val]
    unique_pos_df = pd.DataFrame(with_chips, columns=['item_name - price']).value_counts()
    print (unique_pos_df)

    without_chips = [val + ' - $' + str(price) for val, description, price in zip(df.item_name, df.choice_description, df.item_price) if isNaN(description) and 'Chips' not in val]
    unique_pos_df_nc = pd.DataFrame(without_chips, columns=['item_name - price']).value_counts()
    print (unique_pos_df_nc)
    print('От стоимости бутылки воды не зависит никакая другая стоимость, поэтому в конце предлагается на выбор любая цена соуса/бутылки')

# task1()
# task2()
# task3()
# task4()
# task5()
# task6()
# task7()
# task8()
# task9()
# task10()
# task11()
# task12()
