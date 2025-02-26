## Task 1: Defining your Problem and Audience


## Task 2: Propose a Solution



## Task 3: Dealing with the Data



## Task 4: Building a Quick End-to-End Prototype

https://huggingface.co/spaces/drewgenai/midterm_poc


## Task 5: Creating a Golden Test Data Set


The dataset is based on the submitted documents and the base model performed well across all metrics.

Base model evaluation {'context_recall': 1.0000, 'faithfulness': 1.0000, 'factual_correctness': 0.7540, 'answer_relevancy': 0.9481, 'context_entity_recall': 0.8095, 'noise_sensitivity_relevant': 0.1973}


## Task 6: Fine-Tuning Open-Source Embeddings
Link to fine tuning and testing
https://github.com/drewgenai/midterm_poc/blob/main/03-testembedtune.ipynb
link to fine tuned dataset
https://huggingface.co/drewgenai/midterm-compare-arctic-embed-m-ft



## Task 7: Assessing Performance

I ran the RAGAS evaluation on the finetuned model and the openai model as well. 

The finetuned model performed well across all metrics as well but not quite as well as the base Snowflake/snowflake-arctic-embed-m model where it didn't perform as well in context recall, but slightly in noise sensitivity.

The openai model performed well across all metrics but not as well as the base Snowflake/snowflake-arctic-embed-m model, but with slightly worse noise sensitivity.

Finetuned model {'context_recall': 1.0000, 'faithfulness': 0.8500, 'factual_correctness': 0.7220, 'answer_relevancy': 0.9481, 'context_entity_recall': 0.7917, 'noise_sensitivity_relevant': 0.1111}

Openai model {'context_recall': 1.0000, 'faithfulness': 1.0000, 'factual_correctness': 0.7540, 'answer_relevancy': 0.9463, 'context_entity_recall': 0.8095, 'noise_sensitivity_relevant': 0.3095}

With the results as they are using the Snowflake/snowflake-arctic-embed-m model makes sense for this use case.


## Final Submission

1. GitHub: https://github.com/drewgenai/midterm_poc/blob/main/app.py
2. Public App link: https://huggingface.co/spaces/drewgenai/midterm_poc
3. Public Fine-tuned embeddings: https://huggingface.co/drewgenai/midterm-compare-arctic-embed-m-ft
