# This will be the final exec file for the project

# Importing the required libraries
import rdflib
import pandas as pd
from collections import OrderedDict
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
from sentence_transformers import SentenceTransformer, util
import argparse

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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

## Function to transform the dictionary to handle multiple labels
def transform_dict(original_dict):
    """
    Transforms a dictionary where URLs are the values into a dictionary where
    URLs are the keys and the values are concatenated labels associated with each URL.

    Args:
    original_dict (OrderedDict): The original dictionary with labels as keys and URLs as values.

    Returns:
    dict: A dictionary with URLs as keys and concatenated labels as values.
    """
    new_dict = {}
    for label, url in original_dict.items():
        if url in new_dict:
            new_dict[url] += " " + label  # Concatenating labels with space; change as needed
        else:
            new_dict[url] = label
    return new_dict

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

## Function for Levenshtein distance/similarity
def levenshtein_similarity(str1, str2):
    levenshtein_distance = Levenshtein.distance(str1, str2)
    max_length = max(len(str1), len(str2))
    if max_length == 0:
        return 1.0  # Both strings are empty
    normalized_distance = levenshtein_distance / max_length
    similarity = 1 - normalized_distance
    return similarity

## Function to calculate cosine similarity
def calc_cosine_similarity(str1, str2):
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform([str1, str2])
    return cosine_similarity(count_matrix)[0][1]

## Function to calculate Jaccard similarity
def jaccard_similarity(str1, str2):
    # Tokenize the strings into sets of words
    set1 = set(str1.split())
    set2 = set(str2.split())

    # Find the intersection and union of the two sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate the Jaccard score
    if not union:  # Handle the edge case where both strings might be empty
        return 0.0
    return len(intersection) / len(union)

## Functions for vectorization and cosine similarity
def cosine_vectorize_labels(labels):
    """
    Converts a list of labels into TF-IDF vectors using TfidfVectorizer.

    Args:
    labels (list): List of all labels from both ontologies.

    Returns:
    TfidfVectorizer, scipy.sparse.csr.csr_matrix: The vectorizer and the TF-IDF matrix.
    """
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(labels)
    return vectorizer, count_matrix

def tfidf_vectorize_labels(labels):
    """
    Converts a list of labels into TF-IDF vectors using TfidfVectorizer.

    Args:
    labels (list): List of all labels from both ontologies.

    Returns:
    TfidfVectorizer, scipy.sparse.csr.csr_matrix: The vectorizer and the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(labels)
    return vectorizer, tfidf_matrix

def precompute_cosine_similarities(count_matrix):
    """
    Computes the cosine similarity matrix for all pairs of labels in the count matrix.

    Args:
    count_matrix (scipy.sparse.csr.csr_matrix): The matrix containing the count vectors.

    Returns:
    numpy.ndarray: A 2D array of cosine similarity scores.
    """
    return cosine_similarity(count_matrix)


## Wrapper function for string matching
def execute_string_matching(metric, data1, data2):
    """
    Executes the selected matching metric on the provided data.

    Args:
    metric (str): A single letter representing the metric to use.
                  'Levenshtein' for Levenshtein Distance,
                  'Jaccard' for Jaccard Similarity,
                  'LinkTransformer' for Link Transformer.
    data1, data2 (str): The data strings to compare.

    Returns:
    result: The result of the chosen metric computation.
    """
    if metric == 'Levenshtein':
        return levenshtein_similarity(data1, data2)
    elif metric == 'Jaccard':
        return jaccard_similarity(data1, data2)
    else:
        raise ValueError("Invalid metric selection for string matching.")
    
## Function to get best matches using string matching
def match_ontologies(onto1_dict, onto1_list, onto2_dict, onto2_list, metric):
    try:
        labels_already_tested_labels = {}  # Dict to store which labels have already been tested to avoid infinite loops
        class_results = {}

        if not onto1_list or not onto2_list:  # Ensure the input lists are not empty
            raise ValueError("Ontology lists must not be empty.")

        for label in onto1_list:
            labels_already_tested_labels[label] = []

        onto2_used_classes = {}
        index_dict_label1 = {}
        for index, label1 in enumerate(onto1_list):
            index_dict_label1[label1] = index

        # Initialize similarity matrix based on the metric
        similarity_matrix = None
        if metric == "Cosine" or metric == "TF-IDF":
            all_labels = onto1_list + onto2_list
            vectorizer, matrix = (cosine_vectorize_labels if metric == "Cosine" else tfidf_vectorize_labels)(all_labels)
            similarity_matrix = precompute_cosine_similarities(matrix)

        while onto1_list:
            label1 = onto1_list.pop()
            label_result = [label1, "", "", 0]
            best_score = 0
            already_tested_labels = set(labels_already_tested_labels[label1])

            for index2, label2 in enumerate(onto2_list):
                if label2 not in already_tested_labels:
                    if metric in ["Levenshtein", "Jaccard"]:
                        matching_score = execute_string_matching(metric, label1, label2)
                    elif metric in ["Cosine", "TF-IDF"]:
                        matching_score = similarity_matrix[index_dict_label1[label1], index_dict_label1.get(label2, -1)]
                        if matching_score == None:  # Skip if there's no entry for the label
                            continue

                    if matching_score == 1:
                        best_score = matching_score
                        label_result = [label1, "", label2, best_score]
                        break
                    if matching_score > best_score:
                        best_score = matching_score
                        label_result[2] = label2

            label_result[3] = best_score
            label_with_best_score = label_result[2]
            class_uri = onto1_dict[label1]
            if label_result[3] == 0 and not label_with_best_score:
                class_results[class_uri] = label_result
            else:
                class2_uri = onto2_dict.get(label_with_best_score)
                if class2_uri and class2_uri not in onto2_used_classes:
                    class_results[class_uri] = label_result
                    onto2_used_classes[class2_uri] = class_uri
                    labels_already_tested_labels[label1].append(label_with_best_score)
                elif class2_uri:
                    result_current_class_in_use = onto2_used_classes.get(class2_uri)
                    if best_score > class_results[result_current_class_in_use][3]:
                        class_results[class_uri] = label_result
                        onto2_used_classes[class2_uri] = class_uri
                        old_used_label = class_results[result_current_class_in_use][0]
                        labels_already_tested_labels[old_used_label].append(label_with_best_score)
                        onto1_list.append(old_used_label)  # Re-add for re-evaluation
                        class_results[result_current_class_in_use] = ["", "", "", 0]

        print(f"String matching was successful using the {metric} metric.")
        return class_results, similarity_matrix, index_dict_label1

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}, None, {}
    
## Function for semantic matching with LLM
def calculate_label_similarity_llm(model_name, onto1_dict, onto2_dict):
    """
    Calculate cosine similarity between pairs of labels from two sets and return the results in a dictionary.
    Each key in the dictionary is the class URI from ontology 1, and each value is a list of tuples,
    each containing the label from ontology 2, the class URI from ontology 2, and the similarity score.

    Parameters:
    model_name (str): Name of the Sentence Transformer model to be used.
    onto1_dict (OrderedDict): Dictionary where keys are labels and values are class URIs for the first ontology.
    onto2_dict (OrderedDict): Dictionary where keys are labels and values are class URIs for the second ontology.

    Returns:
    dict: A dictionary with class URIs from the first ontology as keys and lists of tuples (label, class URI, score) from the second ontology as values.
    """
    
    # Initialize logging
    logging.basicConfig(level=logging.ERROR)
    
    # Determine if a GPU is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = SentenceTransformer(model_name, device=device)
        print(f"Using sentence transformer model: {model_name} on device: {device}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # Fallback to a default model if there's an error
        fallback_model_name = 'all-MiniLM-L6-v2'
        model = SentenceTransformer(fallback_model_name, device=device)
        print(f"Using fallback sentence transformer model: {fallback_model_name} on device: {device}")
    
    try:
        onto1_labels, onto1_classes = zip(*onto1_dict.items())
        onto2_labels, onto2_classes = zip(*onto2_dict.items())

        onto1_label_embeddings = model.encode(list(onto1_labels), convert_to_tensor=True, device=device)
        onto2_label_embeddings = model.encode(list(onto2_labels), convert_to_tensor=True, device=device)

        similarity_scores = util.pytorch_cos_sim(onto1_label_embeddings, onto2_label_embeddings)

        # Initialize the dictionary to hold results
        results_dict = {}

        # Fill the dictionary with similarity scores
        for i, onto1_class in enumerate(onto1_classes):
            results_dict[onto1_class] = {}
            for j, onto2_class in enumerate(onto2_classes):
                results_dict[onto1_class][onto2_class] = similarity_scores[i][j].item()

        # Sort the dictionary entries by similarity score within each onto1_class
        sorted_results_dict = {}
        for onto1_class in results_dict:
            sorted_onto2_classes = sorted(results_dict[onto1_class].items(), key=lambda x: x[1], reverse=True)
            sorted_results_dict[onto1_class] = dict(sorted_onto2_classes)

        print(f"LLM similarity calculation was successful using the {model_name} model.")
        return sorted_results_dict

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        return {}

## Functions to get best matches using LLM
def set_new_match(class1_uri, class2_uri, score, onto2_used_classes, class_results):
    """Set a new match for class1_uri and class2_uri."""
    onto2_used_classes[class2_uri] = class1_uri
    class_results[class1_uri] = ["", class2_uri, "", score]  # label is empty for now

def update_matching(new_class1_uri, class2_uri, new_score, old_class1_uri, onto2_used_classes, class_results, onto1_class_list):
    """Update matches when a better score is found."""
    onto2_used_classes[class2_uri] = new_class1_uri
    class_results[old_class1_uri] = ["", "", "", 0]  # Clear old match
    class_results[new_class1_uri] = ["", class2_uri, "", new_score]
    onto1_class_list.append(old_class1_uri)

def reevaluate(class1_uri, class2_uri, already_tested_classes, onto1_class_list):
    """Re-add class1_uri for re-evaluation."""
    onto1_class_list.append(class1_uri)
    already_tested_classes[class1_uri].add(class2_uri)

def perform_matching_llm(dict_similarity_scores_llm):
    # Initialize dictionaries and lists
    already_tested_classes = {}
    class_results = {}
    onto2_used_classes = {}
    onto1_class_list = list(dict_similarity_scores_llm.keys())

    while onto1_class_list:
        class1_uri = onto1_class_list.pop()
        already_tested_classes[class1_uri] = already_tested_classes.get(class1_uri, set())

        # Iterate over each class2_uri and score from the pre-sorted dictionary
        for class2_uri, score in dict_similarity_scores_llm[class1_uri].items():
            if class2_uri not in already_tested_classes[class1_uri]:
                already_tested_classes[class1_uri].add(class2_uri)  # Mark this class2_uri as tested

                if score >= 0.99:  # Check for a perfect match
                    set_new_match(class1_uri, class2_uri, score, onto2_used_classes, class_results)
                    break  # Found a perfect match, skip further checks for this class1_uri

                # If no perfect match, check if it's not already linked
                if class2_uri not in onto2_used_classes:
                    set_new_match(class1_uri, class2_uri, score, onto2_used_classes, class_results)
                    break  # Successfully linked, no need to continue

                # If already linked, check if the new score is better
                elif score > class_results[onto2_used_classes[class2_uri]][3]:
                    old_class1_uri = onto2_used_classes[class2_uri]
                    update_matching(class1_uri, class2_uri, score, old_class1_uri, onto2_used_classes, class_results, onto1_class_list)
                    break  # Updated the link, no need to continue
            else:
                # This class2_uri was already checked, continue to the next
                continue

    return class_results

## Function to add labels to LLM results
def add_labels(data, onto1_label, onto2_label):
    # Iterate through each key and update the list with labels
    for key, values in data.items():
        # First empty string in the list gets replaced by the label from onto1_label if available
        label1 = onto1_label.get(key)
        if label1 is not None:
            values[0] = label1

        # Second empty string in the list gets replaced by the label from onto2_label using the second entry's key if available
        label2_key = values[1]  # The second entry in the list is assumed to be a key for the onto2_label
        label2 = onto2_label.get(label2_key)
        if label2 is not None:
            values[2] = label2

    return data

## Function to check for overlapping matches
def check_for_overlapping_matches(final_matching_results, string_matching_results, matched_results_llm_with_labels):

    for class_name, values in string_matching_results.items():
        class_2 = values[1]
        if class_2 and matched_results_llm_with_labels[class_name][1] == class_2:
            higher_score = values[3]
            label_higher_score = values[2]
            if matched_results_llm_with_labels[class_name][3] > higher_score:
                higher_score = matched_results_llm_with_labels[class_name][3]
                label_higher_score = matched_results_llm_with_labels[class_name][2]
            final_matching_results[class_name] = [values[0], values[1], label_higher_score, higher_score]
    
    return final_matching_results

## Function to remove overlapping matches - !!IN PLACE!!
def remove_overlapping_keys(overlapping_results_keys, string_matching_results, llm_matching_results_with_labels):
    for key in overlapping_results_keys:
        string_matching_results.pop(key, None)
        llm_matching_results_with_labels.pop(key, None)

## Function to calc matches with "other matching approach"
# method to calculate the score for a given dict of matched classes
# this methods enables us to calculate the String matching score for the results of the LLM and vice versa
def calc_score_for_matched_classes(matched_classes, metric, dict_sim_scores_llm={}):
    matches_with_score = {}
    for class_name, values in matched_classes.items():
        label1 = values[0]
        class_2 = values[1]
        label2 = values[2]

        if label2:
            matching_score = 0
            try:
                if metric == "llm":
                    # Ensures that we have nested dictionaries and that class_2 is a valid key under class_name
                    matching_score = dict_sim_scores_llm.get(class_name, {}).get(class_2, 0)
                else:
                    matching_score = execute_string_matching(metric, label1, label2)  # Calculate string matching score
            except Exception as e:
                print(f"An error occurred while calculating scores: {e}")
                matching_score = 0  # Default to 0 if there's an error

            matches_with_score[class_name] = [label1, class_2, label2, matching_score]
        else:
            matches_with_score[class_name] = [label1, class_2, "", 0]

    return matches_with_score

## Function to combine the non-overlapping results of the two matching approaches
# This code sorts the class names of ontology 1 by score in descending order
# This step is important before the combining step as it ensures that first the classes with the highest score
# matches get combined and therefore for these classes less conflict occur as they are handled in the beginning
def sort_ontology_classes_by_score(string_matching_results, string_matches_for_llm, matched_results_llm_with_labels, llm_matches_for_string):
    onto1_class_names_by_score = {}
    
    for onto1_class in string_matching_results:
        entries = [
            string_matching_results[onto1_class],
            string_matches_for_llm[onto1_class],
            matched_results_llm_with_labels[onto1_class],
            llm_matches_for_string[onto1_class]
        ]
        highest_score_entry = max(entries, key=lambda x: x[-1])  # Assuming the score is the last element in each entry
        onto1_class_names_by_score[onto1_class] = highest_score_entry

    # Sorting classes based on the score in descending order
    sorted_classes = sorted(onto1_class_names_by_score.items(), key=lambda x: x[1][3], reverse=True)
    return OrderedDict(sorted_classes)

## Function to resolve conflicts and pick the best matching
# In this code block the four lists are taken and the best matching of the four is picked.
# The code handles conflicts (class2 already used in an earlier match for a different class1)
# It is not perfect as if in the end there are just conflicts that can't be solve these will just be added as empty results
# => but as this are only a few entries and the complexity of handling those will be quite large we will stick with these results for now
def resolve_conflicts_and_pick_best(sorted_onto1_class_names_by_score, string_matching_results, string_matches_for_llm, matched_results_llm_with_labels, llm_matches_for_string):
    remaining_results = {}
    already_used_class2 = []
    conflicts = {}

    for onto1_class in sorted_onto1_class_names_by_score:
        results = [
            string_matching_results[onto1_class],
            string_matches_for_llm[onto1_class],
            matched_results_llm_with_labels[onto1_class],
            llm_matches_for_string[onto1_class]
        ]

        while results:
            highest_score_entry = max(results, key=lambda x: x[-1])
            class2_highest_entry = highest_score_entry[1]
            if class2_highest_entry in already_used_class2:
                results.remove(highest_score_entry)
            else:
                already_used_class2.append(class2_highest_entry)
                remaining_results[onto1_class] = highest_score_entry
                break

        if not results or class2_highest_entry == "":
            remaining_results[onto1_class] = ["", "", "", 0]  # Add empty result for unresolved conflicts

    return remaining_results

## Function to filter results that are above a certain threshold
def filter_results_by_threshold(final_matching_results, threshold):
    final_results_over_threshold = {}
    
    for class_name, values in final_matching_results.items():
        label1, class2, label2, score = values
        if score > threshold:
            final_results_over_threshold[class_name] = values

    return final_results_over_threshold

## Function to take results and write them to an rdf file
def process_results_and_serialize_to_rdf(final_results_over_threshold, filepath="ontology_alignment_results.rdf"):

    # Convert the final results to a list of tuples
    final_matches = []

    for match in final_results_over_threshold:
        class1 = match
        class2 = final_results_over_threshold[match][1]
        score = final_results_over_threshold[match][3]
        final_matches.append((class1, class2, score, "="))

    # Initialize graph
    g = rdflib.Graph()

    # Define namespaces
    KNOWLEDGEWEB = rdflib.Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment#")
    g.bind("kw", KNOWLEDGEWEB)

    # Create the root element for the alignment
    alignment = rdflib.URIRef("http://example.org/alignment")

    # Add basic alignment properties
    g.add((alignment, rdflib.namespace.RDF.type, KNOWLEDGEWEB.Alignment))
    g.add((alignment, KNOWLEDGEWEB.xml, rdflib.Literal("yes")))
    g.add((alignment, KNOWLEDGEWEB.level, rdflib.Literal("0")))
    g.add((alignment, KNOWLEDGEWEB.type, rdflib.Literal("??")))

    # Add each match to the graph
    for entity1, entity2, measure, relation in final_matches:
        cell = rdflib.URIRef(f"http://example.org/cell/{entity1.split('#')[-1]}_{entity2.split('#')[-1]}")
        g.add((cell, rdflib.namespace.RDF.type, KNOWLEDGEWEB.Cell))
        g.add((cell, KNOWLEDGEWEB.entity1, rdflib.URIRef(entity1)))
        g.add((cell, KNOWLEDGEWEB.entity2, rdflib.URIRef(entity2)))
        g.add((cell, KNOWLEDGEWEB.measure, rdflib.Literal(measure, datatype=rdflib.namespace.XSD.float)))
        g.add((cell, KNOWLEDGEWEB.relation, rdflib.Literal(relation)))
        g.add((alignment, KNOWLEDGEWEB.map, cell))

    # Serialize the graph to an RDF file (e.g., in RDF/XML format)
    with open(filepath, "wb") as f:
        f.write(g.serialize(format='pretty-xml').encode("utf-8"))

# Get all user inputs
onto1_path = "test_ontologies/mouse.owl"
onto2_path = "test_ontologies/human.owl"
threshold = 0.8
metric = "Jaccard"
llm = "all-MiniLM-L12-v2"


# Apply the functions here
## Read the ontologies
onto1_graph = load_ontology(onto1_path)
onto2_graph = load_ontology(onto2_path)

## Extract the information from the ontologies
onto1_dict, onto1_list = extract_ontology_details_to_dict(onto1_graph)
onto2_dict, onto2_list = extract_ontology_details_to_dict(onto2_graph) 

## Transform the dictionaries to handle multiple labels (if any) - just needed for LLM post-processing
onto1_transformed_dict = transform_dict(onto1_dict)
onto2_transformed_dict = transform_dict(onto2_dict)

## Apply exact string matching and put into final matching results
exact_matches, onto1_dict_after_exact, onto1_list_after_exact, onto2_dict_after_exact, onto2_list_after_exact = exact_string_match(onto1_dict, onto1_list, onto2_dict, onto2_list)
final_matching_results = exact_matches.copy()

## Apply string matching
string_matching_results, similarity_matrix, index_dict_label1 = match_ontologies(onto1_dict_after_exact, onto1_list_after_exact, onto2_dict_after_exact, onto2_list_after_exact, metric)

## Apply LLM
dict_similarity_scores_llm = calculate_label_similarity_llm(llm, onto1_dict_after_exact, onto2_dict_after_exact)
llm_matching_results = perform_matching_llm(dict_similarity_scores_llm)
llm_matching_results_with_labels = add_labels(llm_matching_results, onto1_transformed_dict, onto2_transformed_dict)

## Check for overlapping matches and remove from original lists
final_matching_results = check_for_overlapping_matches(final_matching_results, string_matching_results, llm_matching_results_with_labels)
remove_overlapping_keys(final_matching_results, string_matching_results, llm_matching_results_with_labels)

## Get other metric for the non-overlapping results
string_matches_for_llm = calc_score_for_matched_classes(llm_matching_results_with_labels,
                                                        metric)
llm_matches_for_string = calc_score_for_matched_classes(string_matching_results,
                                                        "llm",
                                                        dict_sim_scores_llm=dict_similarity_scores_llm)

## Sort the ontology classes by score in descending order
sorted_onto1_class_names_by_score = sort_ontology_classes_by_score(string_matching_results, string_matches_for_llm, llm_matching_results_with_labels, llm_matches_for_string)

## Resolve conflicts and pick the best matching
remaining_results = resolve_conflicts_and_pick_best(sorted_onto1_class_names_by_score, string_matching_results, string_matches_for_llm, llm_matching_results_with_labels, llm_matches_for_string)
final_matching_results.update(remaining_results)

## Filter results by threshold
final_matching_results = filter_results_by_threshold(final_matching_results, threshold)

## Serialize the results to an RDF file
process_results_and_serialize_to_rdf(final_matching_results)