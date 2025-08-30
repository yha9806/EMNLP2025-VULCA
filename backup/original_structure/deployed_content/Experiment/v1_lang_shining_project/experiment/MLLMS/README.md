# Experiment Protocol for Large Language Model Testing

## 1. Experiment Overview

This document outlines the experimental procedure for testing Large Language Models (LLMs). The experiment involves a series of evaluations structured into major and minor rounds.

*   **Total Major Rounds:** 12 (each corresponding to a unique "Monthly Image").
*   **Minor Rounds per Major Round:** 9, consisting of:
    *   8 rounds of Persona-based Question & Answering (Q&A).
    *   1 round of Basic Question & Answering (Q&A).

## 2. Referenced Directories & Files

The experiment relies on the following directory structure and files located under `...\MLLMS\` path:

*   **Images:** `...\MLLMS\images`
    *   Contains the 12 "Monthly Images" used for each major round. Each image may consist of multiple large-format slices.
*   **Persona Cards:** `...\MLLMS\persona`
    *   Contains various persona definitions used in the persona-based Q&A rounds.
*   **Prompt File:** `...\MLLMS\prompt\prompt.md`
    *   Contains the primary prompt template used for initiating Q&A sessions.
*   **Knowledge Base:** `...\MLLMS\knowledge_dataset\knowledge_base.json`
    *   A JSON file containing supplementary information or knowledge relevant to the Q&A tasks.
*   **Output/Feedback Directory:** `...\MLLMS\feedbacks\`
    *   The root directory where all experimental results (feedback from models) are stored. Sub-directories are created per model.

## 3. Experimental Procedure

The experiment proceeds in a cyclical manner through major and minor rounds.

### Major Round Cycle (12 Total)

1.  For each major round, select one of the 12 "Monthly Images" from the `...\MLLMS\images` directory.

### Minor Round Cycle (9 per Major Round)

For each selected "Monthly Image" (major round), complete the following 9 minor rounds:

#### A. Persona-based Q&A (8 Rounds)

For each of the 8 persona-based Q&A rounds:

1.  **Step 1: Input Persona Card:**
    *   Send the content of a selected persona card from the `...\MLLMS\persona` directory.
2.  **Step 2: Upload Image:**
    *   Upload all large-format slices corresponding to the current "Monthly Image".
3.  **Step 3: Input Prompt & Knowledge Base:**
    *   Send the prompt content from `...\MLLMS\prompt\prompt.md`.
    *   Concatenate or provide the content from `...\MLLMS\knowledge_dataset\knowledge_base.json`.
4.  **Step 4: Save Results:**
    *   Record the model's output. The file saving structure is detailed below. (Example reference for structure: `...\MLLMS\feedbacks\Qwen2.5`).

#### B. Basic Q&A (1 Round)

1.  **Step 1: Upload Image:**
    *   Upload all large-format slices corresponding to the current "Monthly Image".
2.  **Step 2: Input Prompt:**
    *   Send the prompt content (presumably from `...\MLLMS\prompt\prompt.md`, potentially without persona-specific modifications or the knowledge base).
3.  **Step 3: Save Results:**
    *   Record the model's output. The file saving structure is detailed below.

### Iteration Logic

After completing all 9 minor rounds for the current image, proceed to the next major round by selecting a new image and repeating the 9 minor rounds. This continues until all 12 major rounds are completed.

## 4. Output File Structure

The results from the experiments are saved with the following hierarchical structure:

*   **Parent Directory for each Model:**
    *   `...\MLLMS\feedbacks\<Model_Name>\`
    *   Example: `...\MLLMS\feedbacks\gemini2.5`

*   **Sub-directory for each Image within a Model's Directory:**
    *   `...\MLLMS\feedbacks\<Model_Name>\<Image_Name>\`

*   **Result File Naming Convention:**
    *   **Persona-based Q&A:** `<Image_Name>+<Persona_Card_Name>.txt`
        *   Example: `DecemberImage+OptimisticPersona.txt`
    *   **Basic Q&A:** `<Image_Name>+(Basic).txt`
        *   Example: `DecemberImage+(Basic).txt`

A concrete example of the file saving structure can be found by referencing the `...\MLLMS\feedbacks\gemini2.5` directory mentioned in the original README.
