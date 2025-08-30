# Human-Centric Artificial Intelligence projects

This repository contains Django implementations of Human-Centric Artificial Intelligence projects: 

---

## 🏗 Project Structure

The project is organized into 5 main parts:

1. **Part 1: Automated Machine Learning - Supervised Learning Interface**  
   - **Purpose:** Provide a web-based interface for supervised learning tasks, enabling users to uplaod datasets, visualize them, and train
   - machine learning models interactively.  
   - **Key features**:
     - 📂 CSV dataset upload  
     - 📊 Visualization of features and labels  
     - ⚙️ Model training with train/test split, hyperparameter tuning, and evaluation  
     - 🔄 User choice between different algorithms and settings  
     - 🖥 Integration with Django interface  

2. **Part 2: Active Learning for Tex Classification**  
   - **Purpose**: Develop an interactive text classification system for seniment analysis on the IMDB 50k dataset, with
   - a focus on active learning. Instead of relying on a large labeled dataset, the system incrementally queries the user (or a simulated annotator)
   - to label the most informative samples, reducing annotation effort while maintaing high performance. 
   - **Key features**:
     - 📝 Train a supervised text classifier (baseline accuracy).
     - 🎯 Implement pool-based active learning strategies.
     - 👩‍💻 Support simulated or real users for labeling.
     - 📈 Track model performance as labeling progresses.
     - 🔄 Optional extensions: batch or stream-based active learning.
  
3. **Part 3: Explainability**  
   - **Purpose**: Provide an interactive tool for exploring model interpretability and counterfactual explanations using the
   - Palmer Penduins dataset. The app highlights the trade-off between accuracy and complexity while helping users understand and question
   - model predictions. 
   - **Key features**:
     - 🌳 Train interpretable models (decision trees, logistic regression) with adjustable complexity.
     - 🎚 Interactive slider to control sparsity vs. accuracy.
     - 📊 Display accuracy, model size (leaves/features), and structure.
     - 🔄 Generate counterfactual explanations for selected samples.
     - 👩‍💻 User interface for exploring predictions and alternatives.
  
4. **Part 4: Influence of future predictions over active learning of users' tastes for recommender systems**  
   - **Purpose**: Design an interactive recoomender system that tackles the cold start problem by learning
   - user preferences through guided input and showing how their ratings affect recommendations. 
   - **Key features**:
     - 🎬 Use the MovieLens dataset for movie recommendations.
     - 📝 Guided active learning: users rate movies while seeing how their input impacts the system.
     - 👩‍🔬 Design and document a user study to test hypotheses about the method.
     - 🖥 Build an interactive interface for the study with a landing page, PDF documentation, and study launch.
     - ⚙️ Implement recommendations using matrix factorization with user/item embeddings.
  
5. **Part 5: Reinforcement Learning with Human Feedback**  
   - **Purpose**: Develop a reinforcement learning system where a mouse learns to navigate a grid-world and
   - collect cheese, refined by human feedback to prefer desirable outcomes and avoid traps. 
   - **Key features**:
     - 🐭 Grid-world game environment with cheese, traps, and walls.
     - ⚡ Learn policies using the REINFORCE algorithm and neural networks.
     - 👩‍💻 Incorporate human feedback (or simulated feedback) to refine the learned policy.
     - 📊 Estimate rewards from feedback using the Bradley-Terry model.
     - 🔄 Retrain the policy network with the refined reward while maintaining baseline behavior.

---

## ⚙️ Installation

### Requirements
- Install **Docker** and **Docker Compose** on your system
  - Docker Desktop for Mac/Windows
  - Linux: sudo apt install docker.io docker-compose
 
### Steps
1. **Clone the repository**:
     git clone [https://github.com/your-username/your-project.git](https://github.com/sevenxx07/Human-CentricAI_Project.git)
     cd your-project
2. **Build the Docker image**:
     docker-compose build
   - This uses the Dockerfile in docker/Dockerfile to create an environment with all dependencies

3. **Run the project**:
   docker-compose up
   - The command automatically runs:
     - python manage.py makemigrations
     - python manage.py migrate
     - python manage.py runserver 0.0.0.0:8000
  - Your Django app will be accessible at: http://127.0.0.1:8000

4. **Stop the project**:
   docker-compose down


# Create a superuser (optional)
python manage.py createsuperuser
