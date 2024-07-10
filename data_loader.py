import os
import fitz
import pymupdf4llm
import re
import logging
import yaml
import json
from langchain_text_splitters import NLTKTextSplitter

logger = logging.getLogger(__name__)

with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

# Convert PDFs to Markdown
def convert_pdfs_to_markdown(directory):
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        md_text = pymupdf4llm.to_markdown(pdf_file)
        markdown_file = os.path.splitext(pdf_file)[0] + ".md"
        
        # Write the markdown text to a file with the same name as the PDF
        with open(markdown_file, "w", encoding="utf-8") as output:
            output.write(md_text)

# Delete unwanted sections
def filter_sections(all_sections):
    logger.info("Filtering sections")
    
    unwanted_sections = [section for section in all_sections if any(text in re.search(r'#{1,6} (.*)', section, re.IGNORECASE).group(1)  for text in config['unwanted_sections_header'])]
    filtered_sections = [section for section in all_sections if section not in unwanted_sections]

    return filtered_sections

# Split markdown text by section
def split_markdown_by_section(markdown_text):
    # Lowercase the markdown text to make the regex case-insensitive
    markdown_text = markdown_text.lower()

    # Delete figures and images
    markdown_text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
    
    # Split by markdown headers
    sections = re.split(r'(#{1,6} .*)', markdown_text)  
    result = []
    for i in range(1, len(sections), 2):
        header = sections[i]
        content = sections[i + 1] if i + 1 < len(sections) else ""
        result.append(header + content)
    return result

# Save sections to a list
def save_sections_to_list(directory):
    sections_list = []
    markdown_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.md')]
    token_splitter = NLTKTextSplitter(chunk_size=3000)

    for markdown_file in markdown_files:
        try:
            with open(markdown_file, 'r', encoding='utf-8') as file:
                markdown_text = file.read()
                sections = split_markdown_by_section(markdown_text)
                filtered_sections = filter_sections(sections)
                
                for section in filtered_sections:
                    chunks = token_splitter.split_text(section)
                    sections_list.extend(chunks)
                    
        except Exception as e:
            logging.error(f"Error processing file {markdown_file}: {e}")
    
    return sections_list

# Process all subfolders
def data_loader_subfolders(main_directory):
    all_sections = []
    subfolders = [os.path.join(main_directory, d) for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    
    for subfolder in subfolders:
        logger.info(f"Loading data from {subfolder}")
        #convert_pdfs_to_markdown(subfolder)
        sections = save_sections_to_list(subfolder)
        all_sections.extend(sections)
    
    with open("documents.json", "w") as file:
        json.dump(all_sections, file)
    
    return all_sections