import subprocess
import pandas as pd

# Define the configurations
configurations = [
    {'metric': 'Jaccard', 'threshold': 0.6},
    {'metric': 'Jaccard', 'threshold': 0.7},
    {'metric': 'Jaccard', 'threshold': 0.8},
    {'metric': 'Jaccard', 'threshold': 0.9},
    {'metric': 'Levenshtein', 'threshold': 0.6},
    {'metric': 'Levenshtein', 'threshold': 0.7},
    {'metric': 'Levenshtein', 'threshold': 0.8},
    {'metric': 'Levenshtein', 'threshold': 0.9},
    {'metric': 'Cosine', 'threshold': 0.6},
    {'metric': 'Cosine', 'threshold': 0.7},
    {'metric': 'Cosine', 'threshold': 0.8},
    {'metric': 'Cosine', 'threshold': 0.9},
    {'metric': 'TF-IDF', 'threshold': 0.6},
    {'metric': 'TF-IDF', 'threshold': 0.7},
    {'metric': 'TF-IDF', 'threshold': 0.8},
    {'metric': 'TF-IDF', 'threshold': 0.9}
]

# Store results
results = []

# Run configurations
for config in configurations:
    command = [
        'python', 'ontology-alignment.py', 
        '--onto1_path', 'test_ontologies/mouse.owl',
        '--onto2_path', 'test_ontologies/human.owl',
        '--threshold', str(config['threshold']),
        '--metric', config['metric'],
        '--output', 'ontology_alignment_results_test.rdf'
    ]
    # Capture the output
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    output = result.stdout
    
    # Parse output
    times = {}
    for line in output.splitlines():
        if "Time" in line:
            key, value = line.split(":")
            times[key.strip()] = float(value.strip())
    
    # Append results with times
    times.update({
        'Metric': config['metric'],
        'Threshold': config['threshold']
    })
    results.append(times)

    print("Finished configuration:", config)

# Create DataFrame
df = pd.DataFrame(results)
print(df)
# Optionally save to CSV
df.to_csv('execution_times.csv', index=False)
