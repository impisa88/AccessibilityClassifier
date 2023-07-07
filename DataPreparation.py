import pandas as pd
import collections


class Preparation:

       accomodation = pd.read_csv("accomodation.csv")
       tour = pd.read_csv("tour.csv")
       restaurant = pd.read_csv("restaurant.csv")
       convinence = pd.read_csv("convinence.csv")

       temp = accomodation._append([tour, restaurant, convinence])

       data = temp.drop(columns=['Name', 'District', 'Address', 'Contact', 'Homepage', 'Upvote','Disabled Lodging Facility', 'Disabled Ticket Office',
              'Service for the visually-impaired', 'Service for the hearing-impaired', 'Information Service', 'Wheelchair Rental'])

       listData = []

       for i in data.values:
              cont = collections.Counter(i)
              if cont['Y'] >= cont['N']:
                     listData.append(1)
              else:
                     listData.append(0)


       temp.insert(1, "Access", listData, True)

       temp.to_csv('finalDataset.csv')

Preparation();