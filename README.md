### How to Deploy your Streamlit App using Streamlit Sharing
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/kingard/stock-prediction-app/app.py)

### Steps
- Research on [google colab](https://colab.research.google.com/). Here google provides a remote jupyter notbook and adequate processing capacity (GPUs) for creating ML models
- Save you model in a `.h5`. This is a Hierarchical Data Format (HDF) that can store multidimensional arrays of scientific data. With the model saved in this file, we can add the file to te same folder as our streamlit app so that we can build a feature based that consumes the model
- Create your app folder and create a virtual environment

    **create**: `Python3 -m venv venv`
	
    **activate**: 
	
    for windows: ` venv/Scripts/activate`
	
    for mac users: `source venv/bin/activate`

- Install requirements: `pip install -r requirements.txt`
- run app: `streamlit run app.py`
- Deployment

  You need 

        - Github Account: where you will host the app
        - share.streamlit.io: create an account with github and signup with the same email used in github
        - App + Requirements.txt

### Extras:
Creating your own Requirements.txt file

	- Pipenv  pipfile
	- pip freeze > requirements.txt
	- pipreqs /home/project/location



#### Integrating the Play Button 
With this format you can access your app directly from github on the README file just like the streamlit badge at the top of this file. 

```[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourGitHubName/yourRepo/yourApp/)```