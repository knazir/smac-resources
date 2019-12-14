# SmAC: A Smart Anti-Cheat Tool for Multiplayer Online Games

These are the supporting files used for data processing and training machine learning models for this CS221 project on detecting cheating in multiplayer online games.

## Structure

### api
* Contains the player beahvior collection server.
* To run, Node.js and NPM must be installed. You can `npm install && npm start` inside the `api` directory to start the server.


### stray
* Contains data files that were in a format usable by R to run the [stray](https://github.com/pridiltal/stray) algorithm, which can be installed following the instructions on their GitHub repostiroy.
* Sample commands used to find and visualize anomalies can be found in `stray.r`.


### svm
* Contains raw game state data and Python scripts used to preprocess the data and train an SVM classifier.
* The `preprocess.py` script will compute game state gradients
* The `svm.py` script will train an SVM using SKLearn on the training data output from preprocessing.
