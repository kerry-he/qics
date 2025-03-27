:layout: landing

.. raw:: html

    <style>
        .full-width-banner {
            width: 100vw;
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
            padding: 4rem 0;
            background: linear-gradient(0deg,
                rgba(136, 180, 167, 0.5) 50%, 
                transparent 100%
            );
        }


        html {
            overflow-x: hidden;
            width: 100%;
        }

        .full-width-banner h1 {
            font-family: "Poppins", sans-serif;
            font-weight: 600;
            font-style: normal;
            color: #36454fff; /* default for light mode */
        }

        html.dark .full-width-banner h1 {
            color: #88b4a7ff; /* dark mode color */
        }

        .full-width-banner p {
            font-family: "Poppins", sans-serif;
            font-weight: 300;
            font-style: normal;
            color: #36454fff; /* default for light mode */
        }

        html.dark .full-width-banner p {
            color: #88b4a7ff; /* dark mode color */
        }

        .button {
            margin-left: 5px;
            margin-right: 5px; /* Adds 10px space to the right of the button */
        }

        a.button.reference.internal:hover {
            color: #88b4a7ff;   
        }

        a.button.reference.external {
            /* Additional external link styling */
            background-color: #36454fff;
            color: #ffffff;
            border: 2px solid #36454fff;
        }

        a.button.reference.external:hover {
            background-color: #ffffff;
            color: #36454fff;   
            border:2px solid #36454fff;
        }

        html.dark a.button.reference.external:hover {
            background-color: #000000;
        }

        .features {
            display: flex;
            justify-content: space-around;
            align-items: flex-start;
            gap: 2rem;
            margin: 2rem 0;
        }

        .features-section {
            text-align: center;
            margin: 4rem 0;
        }

        .features-section h2 {
            font-family: "Poppins", sans-serif;
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }

        .feature {
            flex: 1;
            text-align: center;
            padding: 1rem;
        }
        .feature-logo {
            max-width: 80px; /* Adjust size as needed */
            margin-bottom: 1rem;
        }
        .feature h3 {
            font-family: "Poppins", sans-serif;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .feature p {
            font-family: "Poppins", sans-serif;
            font-weight: 300;
            margin-bottom: 0.5rem;
            font-size: 0.8rem;
        }
    </style>

    <head>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap" rel="stylesheet">
    </head>

    <div class="full-width-banner" style="padding-top: 0px; margin-bottom: 48px; height: 675px;">
        <div class="hero-content">
            <img src="_static/logo.svg" alt="Description">
            <h1 style="font-size: 8rem; margin-bottom: 0px">QICS</h1>
            <p style="font-size: 1.5rem;">Quantum Information Conic Solver</p>
            <div class="container buttons" style="margin-top: 64px">
                <a href="introduction.html" class="button reference internal">Docs</a>
                <a href="https://github.com/kerry-he/qics" class="button reference external">GitHub</a>
            </div>
        </div>
    </div>

    <div class="features-section">
        <h2>Features</h2>
        <div class="features">
            <div class="feature">
                <h3>Quantum entropy programming</h3>
                <p>Supports quantum relative entropies, (sandwiched) Rényi entropies, matrix geometric means, and more.</p>
            </div>
            <div class="feature">
                <h3>Semidefinite programming</h3>
                <p>Achieves comparable performance to state-of-the-art semidefinite programming software.</p>
            </div>
            <div class="feature">
                <h3>Complex-valued matrices</h3>
                <p>Hermitian matrices are directly supported without needing to reformulate the problem.</p>
            </div>
        </div>
    </div>


.. toctree::
   :hidden:
   :maxdepth: 3

   Introduction<introduction.rst>
   guide/index.rst
   examples/index.rst
   api/index.rst
