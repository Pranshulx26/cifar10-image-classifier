<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIFAR-10 Image Classifier Web App</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom styles for better readability and aesthetics */
        body {
            font-family: 'Inter', sans-serif;
            /* Adjusted background color for a softer look */
            background-color: #ebf0ef; /* A very light, subtle greenish-gray */
            color: #000; /* Changed to black for maximum visibility */
            line-height: 1.6;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            /* Adjusted background for the container */
            background-color: #fdfdfd; /* A subtle off-white */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: 700;
        }
        h1 { font-size: 2.5rem; }
        h2 { font-size: 2rem; }
        h3 { font-size: 1.75rem; }
        h4 { font-size: 1.5rem; }
        p {
            margin-bottom: 1em;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        code {
            background-color: #e0e0e0;
            padding: 0.2em 0.4em;
            border-radius: 0.25rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
        }
        pre {
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin-bottom: 1em;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1em;
            border-radius: 0.5rem;
            overflow: hidden; /* Ensures rounded corners apply to content */
        }
        th, td {
            border: 1px solid #ddd;
            padding: 0.75rem;
            text-align: left;
        }
        th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e9e9e9;
        }
        ul {
            list-style-type: disc;
            margin-left: 1.5em;
            margin-bottom: 1em;
        }
        ol {
            list-style-type: decimal;
            margin-left: 1.5em;
            margin-bottom: 1em;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 1em;
            margin-left: 0;
            font-style: italic;
            color: #333; /* Changed to a darker gray for better visibility */
        }
        hr {
            border: 0;
            height: 1px;
            background-color: #ccc;
            margin: 2em 0;
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container bg-white shadow-lg rounded-lg p-6 md:p-10 my-8">
        <h1 class="text-4xl font-extrabold text-center mb-6">
            🧠 CIFAR-10 Image Classifier Web App (PyTorch + Flask)
        </h1>
        <p class="text-lg text-center mb-8">
            An end-to-end deep learning project where we trained a Convolutional Neural Network (CNN) on the CIFAR-10 dataset to classify real-world images into 10 categories. The final model achieves <strong class="text-green-600">86% test accuracy</strong>, and we deployed it as a <strong class="text-blue-600">Flask web application</strong> with support for custom image uploads and predictions.
        </p>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="web-app-video-demo">
            ▶️ Web App Video Demo
        </h2>
        <blockquote class="text-gray-600 mb-6">
            Watch a quick demonstration of the web application in action!
        </blockquote>

        <div class="flex justify-center mb-8">
            <video controls class="w-full max-w-2xl rounded-lg shadow-md" poster="https://placehold.co/640x360/E0E0E0/333333?text=Video+Placeholder">
                <source src="static\demo.mp4" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="table-of-contents">
            📌 Table of Contents
        </h2>
        <ul class="list-disc list-inside mb-8 text-blue-700">
            <li><a href="#overview" class="hover:underline">📊 Overview</a></li>
            <li><a href="#project-goals" class="hover:underline">🧠 Project Goals</a></li>
            <li><a href="#dataset" class="hover:underline">📁 Dataset</a></li>
            <li><a href="#tools-technologies" class="hover:underline">🛠️ Tools & Technologies</a></li>
            <li><a href="#experiments-model-evolution" class="hover:underline">🧪 Experiments & Model Evolution</a></li>
            <li><a href="#final-model-architecture" class="hover:underline">🚀 Final Model Architecture</a></li>
            <li><a href="#web-app-video-demo" class="hover:underline">▶️ Web App Video Demo</a></li> <!-- Updated link -->
            <li><a href="#key-learnings" class="hover:underline">💡 Key Learnings</a></li>
            <li><a href="#installation" class="hover:underline">📦 Installation</a></li>
            <li><a href="#future-improvements" class="hover:underline">🧩 Future Improvements</a></li>
            <li><a href="#acknowledgements" class="hover:underline">🙌 Acknowledgements</a></li>
        </ul>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="overview">
            📊 Overview
        </h2>
        <p class="mb-4">
            This project aims to solve the image classification problem using the CIFAR-10 dataset. We started with simple models and iteratively built better-performing CNN architectures through experiments, learning rate tuning, regularization, and augmentation. Our final model, <code class="bg-gray-200">CIFAR10ModelV4</code>, achieved <strong class="text-green-600">~86% accuracy</strong> and was deployed using <strong class="text-blue-600">Flask</strong> for interactive prediction.
        </p>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="project-goals">
            🧠 Project Goals
        </h2>
        <ul class="list-disc list-inside mb-8">
            <li>Build a deep learning model to classify CIFAR-10 images.</li>
            <li>Learn about architecture design, loss functions, and optimizers.</li>
            <li>Improve performance through tuning and data augmentation.</li>
            <li>Deploy the trained model on a Flask web app for real-world use.</li>
        </ul>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="dataset">
            📁 Dataset
        </h2>
        <p class="mb-2">
            <strong>Dataset</strong>: CIFAR-10 <br>
            <strong>Source</strong>: <code class="bg-gray-200">torchvision.datasets.CIFAR10</code>
        </p>
        <h3 class="text-2xl font-semibold mb-2">Classes:</h3>
        <ul class="list-disc list-inside mb-4">
            <li>airplane</li>
            <li>automobile</li>
            <li>bird</li>
            <li>cat</li>
            <li>deer</li>
            <li>dog</li>
            <li>frog</li>
            <li>horse</li>
            <li>ship</li>
            <li>truck</li>
        </ul>
        <p class="mb-8">
            Each image is RGB with a shape of <code class="bg-gray-200">(3, 32, 32)</code>.
        </p>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="tools-technologies">
            🛠️ Tools & Technologies
        </h2>
        <div class="overflow-x-auto mb-8">
            <table class="min-w-full bg-white rounded-lg shadow-md">
                <thead>
                    <tr>
                        <th class="py-3 px-4 bg-gray-700 text-white rounded-tl-lg">Purpose</th>
                        <th class="py-3 px-4 bg-gray-700 text-white rounded-tr-lg">Tools/Frameworks</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td class="py-2 px-4">Programming</td><td class="py-2 px-4">Python</td></tr>
                    <tr><td class="py-2 px-4">Deep Learning</td><td class="py-2 px-4">PyTorch, Torchvision</td></tr>
                    <tr><td class="py-2 px-4">Web Framework</td><td class="py-2 px-4">Flask</td></tr>
                    <tr><td class="py-2 px-4">Web UI</td><td class="py-2 px-4">HTML, CSS, JavaScript</td></tr>
                    <tr><td class="py-2 px-4">Visualization</td><td class="py-2 px-4">Matplotlib, mlxtend, Torchmetrics</td></tr>
                    <tr><td class="py-2 px-4">Hosting</td><td class="py-2 px-4">GitHub Pages (for repo), Flask (local app)</td></tr>
                    <tr><td class="py-2 px-4">Environment</td><td class="py-2 px-4">Conda</td></tr>
                </tbody>
            </table>
        </div>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="experiments-model-evolution">
            🧪 Experiments & Model Evolution
        </h2>

        <h3 class="text-2xl font-semibold mb-3">
            1. 🧱 Baseline Fully Connected Model (<code class="bg-gray-200">CIFAR10ModelV0</code>)
        </h3>
        <ul class="list-disc list-inside mb-6">
            <li>Flattened input, two <code class="bg-gray-200">Linear</code> layers</li>
            <li>No activation</li>
            <li>Result: <strong class="text-red-600">~37% accuracy</strong></li>
            <li>Issue: Lost spatial information</li>
        </ul>

        <h3 class="text-2xl font-semibold mb-3">
            2. 🔁 Added ReLU Activations (<code class="bg-gray-200">CIFAR10ModelV1</code>)
        </h3>
        <ul class="list-disc list-inside mb-6">
            <li>Added non-linear layers</li>
            <li>Learning rate too high (0.1) → performance dropped</li>
            <li>Result: <strong class="text-red-600">~33% accuracy</strong></li>
        </ul>

        <h3 class="text-2xl font-semibold mb-3">
            3. 🧠 TinyVGG CNN (<code class="bg-gray-200">CIFAR10ModelV2</code>)
        </h3>
        <ul class="list-disc list-inside mb-6">
            <li>Basic CNN with 2 Conv blocks</li>
            <li>Result: <strong class="text-yellow-600">~56% accuracy</strong></li>
            <li>Improvement via convolutional feature extraction</li>
        </ul>

        <h3 class="text-2xl font-semibold mb-3">
            4. 🔧 Learning Rate Tuning
        </h3>
        <ul class="list-disc list-inside mb-6">
            <li>Changed <code class="bg-gray-200">lr</code> from <code class="bg-gray-200">0.1</code> to <code class="bg-gray-200">0.01</code></li>
            <li>Trained for 20 epochs instead of 5</li>
            <li>Accuracy improved to <strong class="text-yellow-600">~60%</strong></li>
        </ul>

        <h3 class="text-2xl font-semibold mb-3">
            5. 🌀 Data Augmentation
        </h3>
        <ul class="list-disc list-inside mb-6">
            <li>Added <code class="bg-gray-200">RandomHorizontalFlip</code>, <code class="bg-gray-200">ColorJitter</code>, <code class="bg-gray-200">RandomCrop</code></li>
            <li>Helped reduce overfitting and generalize better</li>
        </ul>

        <h3 class="text-2xl font-semibold mb-3">
            6. 🔬 Learning Rate Sweep
        </h3>
        <ul class="list-disc list-inside mb-6">
            <li>Trained for 3 epochs on multiple LRs (<code class="bg-gray-200">1e-1</code>, <code class="bg-gray-200">1e-2</code>, <code class="bg-gray-200">1e-3</code>)</li>
            <li>Chose LR that dropped loss fastest</li>
        </ul>

        <h3 class="text-2xl font-semibold mb-3">
            7. 🧠 Advanced Deep CNN (<code class="bg-gray-200">CIFAR10ModelV4</code>)
        </h3>
        <ul class="list-disc list-inside mb-6">
            <li>4 Conv blocks with:
                <ul class="list-circle list-inside ml-4">
                    <li>Batch Normalization</li>
                    <li>ReLU activations</li>
                    <li>Dropout regularization</li>
                </ul>
            </li>
            <li>Final classifier with:
                <ul class="list-circle list-inside ml-4">
                    <li>Linear → ReLU → Dropout → Linear</li>
                </ul>
            </li>
            <li>Result: <strong class="text-green-600">~86% accuracy on test set</strong></li>
        </ul>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="final-model-architecture">
            🚀 Final Model Architecture
        </h2>
        <pre><code class="language-text">Conv2d(3 → 32) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
Conv2d(32 → 64) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
Conv2d(64 → 128) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
Conv2d(128 → 256) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
→ Flatten → Linear(1024 → 512) → ReLU → Dropout → Linear(512 → 10)
</code></pre>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="web-app-demo">
            🖼 Web App Demo
        </h2>
        <ul class="list-disc list-inside mb-6">
            <li>Upload image via browser</li>
            <li>Prediction made using <code class="bg-gray-200">model_9</code> (CIFAR10ModelV4)</li>
            <li>Responsive UI using HTML/CSS/JS</li>
            <li>Simple REST API via Flask</li>
        </ul>
        <h3 class="text-2xl font-semibold mb-3">To run locally:</h3>
        <pre><code class="language-bash">python app.py
</code></pre>
        <h3 class="text-2xl font-semibold mb-3">Navigate to:</h3>
        <pre><code class="language-text">http://127.0.0.1:5000
</code></pre>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="key-learnings">
            💡 Key Learnings
        </h2>
        <ul class="list-disc list-inside mb-6">
            <li>CNNs significantly outperform linear models for vision tasks</li>
            <li>BatchNorm and Dropout greatly improve generalization</li>
            <li>Data augmentation is essential to avoid overfitting</li>
            <li>Learning rate tuning is critical</li>
            <li>Flask makes model deployment simple and interactive</li>
        </ul>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="installation">
            📦 Installation
        </h2>
        <h3 class="text-2xl font-semibold mb-3">Clone the repo</h3>
        <pre><code class="language-bash">git clone https://github.com/yourusername/cifar10-image-classifier.git
cd cifar10-image-classifier
</code></pre>
        <h3 class="text-2xl font-semibold mb-3">Create environment & install requirements</h3>
        <pre><code class="language-bash">conda create -n cifar10env python=3.10 -y
conda activate cifar10env
pip install -r requirements.txt
</code></pre>
        <h3 class="text-2xl font-semibold mb-3">Run the web app</h3>
        <pre><code class="language-bash">python app.py
</code></pre>
        <h3 class="text-2xl font-semibold mb-3">Then visit:</h3>
        <pre><code class="language-text">http://127.0.0.1:5000
</code></pre>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="future-improvements">
            🧩 Future Improvements
        </h2>
        <ul class="list-disc list-inside mb-6">
            <li>Add Grad-CAM to visualize model decisions</li>
            <li>Add upload preview and confidence scores</li>
            <li>Support images of any size via resizing</li>
            <li>Deploy live using Render or HuggingFace Spaces</li>
        </ul>

        <hr class="my-8">

        <h2 class="text-3xl font-bold mb-4" id="acknowledgements">
            🙌 Acknowledgements
        </h2>
        <ul class="list-disc list-inside mb-6">
            <li>Daniel Bourke's PyTorch Course (ZTM)</li>
            <li>CIFAR Dataset</li>
            <li>Built using 🧠, 🐍, and ☕</li>
        </ul>
    </div>
</body>
</html>
