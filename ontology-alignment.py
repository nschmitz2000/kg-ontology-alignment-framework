# This will be the final exec file for the project

# Importing the required libraries
import rdflib
import pandas as pd
from collections import OrderedDict

# Define all functions here

## Function to load the ontology
def load_ontology(file_path):
    """
    Loads an ontology from a given file path, which can be in RDF (.rdf) or OWL (.owl) format.
    
    Args:
    file_path (str): The file path to the ontology file.
    
    Returns:
    rdflib.Graph: A graph containing the ontology data.
    """
    # Create a new RDF graph
    graph = rdflib.Graph()

    # Bind some common namespaces to the graph
    namespaces = {
        "rdf": rdflib.namespace.RDF,
        "rdfs": rdflib.namespace.RDFS,
        "owl": rdflib.namespace.OWL,
        "xsd": rdflib.namespace.XSD
    }
    for prefix, namespace in namespaces.items():
        graph.namespace_manager.bind(prefix, namespace)

    # Attempt to parse the file
    try:
        graph.parse(file_path, format=rdflib.util.guess_format(file_path))
        print(f"Successfully loaded ontology from {file_path}")
    except Exception as e:
        print(f"Failed to load ontology from {file_path}: {e}")
        return None

    return graph

## Function to preprocess labels
def preprocess_label(label):
    return str(label).replace("_", " ").strip(" ,.").lower()

## Function to load ontology information into dict
def extract_ontology_details_to_dict(graph):
    # Query for classes
    class_query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?class ?label ?label_dt ?label_lang
    WHERE {
        ?class rdf:type owl:Class.
        OPTIONAL { ?class rdfs:label ?label. BIND(datatype(?label) AS ?label_dt) BIND(lang(?label) AS ?label_lang) }
    }
    """
    classes = graph.query(class_query)
    ontology_labels_dict = OrderedDict()
    labels_list = []

    # Process class results
    for row in classes:
        class_uri, label, label_dt, label_lang = row
        class_key = str(class_uri)
        label_str = preprocess_label(label)

        if label_str is None or label_str == "none":
            continue

        if label_str not in ontology_labels_dict:
            ontology_labels_dict[label_str] = class_key
            labels_list.append(label_str)
            
    return ontology_labels_dict, labels_list

## Function to find exact matches
def exact_string_match(onto1_dict, onto1_list, onto2_dict, onto2_list):
    exact_matches = {}
    matched_labels1 = set()
    matched_labels2 = set()

    for label1 in onto1_list:
        for label2 in onto2_list:
            if label1 == label2:
                # Creating the formatted match entry
                class1 = onto1_dict[label1]
                class2 = onto2_dict[label2]
                exact_matches[class1] = [label1, class2, label2, 1]
                
                # Tracking matched labels for later removal
                matched_labels1.add(label1)
                matched_labels2.add(label2)
                
    # Remove matched labels from lists and dictionaries
    for label in matched_labels1:
        onto1_list.remove(label)
        del onto1_dict[label]

    for label in matched_labels2:
        onto2_list.remove(label)
        del onto2_dict[label]

    return exact_matches, onto1_dict, onto1_list, onto2_dict, onto2_list

## Function 

# Function for LLM implementation

# Get all user inputs

# Apply the functions here
