# Covid19-Fake-News-Classfier
A web app using Flask backend to use NLP to detect fake news on Covid19. Achieved 89% accuracy on validation dataset.

**Folder Structure**
1. Static folder: This folder contains CSS files.
2. Templates folder: This folder contains the HTML files. These HTML files will be rendered on the web browser.
3. app.py: This file contains the Flask specific code.
4. model2.sav : trained model
5. usersDB.sqlite3 : Database File containing details of registered Users.

**To Run The Application on localhost :**
- Change current directory to the directory containing the files.
- Install all dependencies
```
pip install -r requirements.txt
```
- Run 'python app.py' on cmd. This will now run your python application using Flask on your local machine. 
- Enter the local host http address into browser.
Thats it ^_^



**Screenshots**


![Home Page](https://github.com/Kakarot-2000/Covid19-Fake-News-Classfier/blob/ux-changes/screenshots/Screenshot%20(341).png?raw=true)
![Login Page](https://github.com/Kakarot-2000/Covid19-Fake-News-Classfier/blob/master/screenshots/Screenshot%20(376).png)
![Predict Page](https://github.com/Kakarot-2000/Covid19-Fake-News-Classfier/blob/ux-changes/screenshots/Screenshot%20(339).png?raw=true)
![Results Page](https://github.com/Kakarot-2000/Covid19-Fake-News-Classfier/blob/ux-changes/screenshots/Screenshot%20(340).png?raw=true)
