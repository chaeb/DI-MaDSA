from sklearn.metrics.pairwise import cosine_similarity

def call_questions():
    return {'depressed':["How often feeling down, depressed, irritable, or hopeless in the past two weeks?"],
            'interest':["How often little interest or pleasure in doing things in the past two weeks?"],
            'sleep':["How often trouble falling or staying asleep, or sleeping too much in the past two weeks?"],
            'tired':["How often feeling tired or having little energy in the past two weeks?"],
            'appetite':["How often poor appetite or overeating in the past two weeks?"],
            'failure': ["How often feeling bad about yourself or that you are a failure in the past two weeks?", "How often have you let yourself or your family down in the past two weeks?"],
            'concentrating': ["How often have you had trouble concentrating on things in the past two weeks?"],
            'moving': ["How often moving or speaking so slowly that other people could have noticed in the past two weeks?"]}

def calculate_similarity(text1,text2):
    return cosine_similarity(text1, text2)[0][0]
