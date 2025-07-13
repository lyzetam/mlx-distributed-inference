#!/usr/bin/env python3
"""
Convert Home Assistant voice command dataset to MLX training format for Gemma.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_homeassistant_data(file_path: str) -> Dict[str, Any]:
    """Load the Home Assistant dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_action_json(intent: str, entities: List[Dict[str, Any]]) -> str:
    """Create a structured action JSON for the assistant response."""
    action_data = {
        "intent": intent,
        "entities": {}
    }
    
    for entity in entities:
        action_data["entities"][entity["entity"]] = entity["value"]
    
    return json.dumps(action_data, indent=2)


def generate_response(intent: str, entities: List[Dict[str, Any]], responses: Dict[str, List[str]]) -> str:
    """Generate a natural language response based on intent and entities."""
    if intent not in responses:
        return f"I'll help you with that {intent.lower()} request."
    
    # Get a response template
    templates = responses[intent]
    template = random.choice(templates)
    
    # Replace placeholders with entity values
    entity_dict = {e["entity"]: e["value"] for e in entities}
    
    try:
        response = template.format(**entity_dict)
    except KeyError:
        # Fallback if template has missing keys
        response = templates[0]
        for entity, value in entity_dict.items():
            response = response.replace(f"{{{entity}}}", value)
    
    return response


def augment_command(text: str, add_please: bool = False, add_can_you: bool = False) -> str:
    """Add variations to commands for robustness."""
    if add_please and random.random() > 0.5:
        if random.random() > 0.5:
            text = f"please {text}"
        else:
            text = f"{text} please"
    
    if add_can_you and random.random() > 0.5:
        text = f"can you {text}"
    
    return text


def create_training_example(
    command: str,
    intent: str,
    entities: List[Dict[str, Any]],
    responses: Dict[str, List[str]]
) -> Dict[str, List[Dict[str, str]]]:
    """Create a single training example in the format expected by MLX."""
    
    # Generate the action JSON
    action_json = create_action_json(intent, entities)
    
    # Generate natural language response
    nl_response = generate_response(intent, entities, responses)
    
    # Combine into assistant response
    assistant_response = f"{nl_response}\n\n<action>\n{action_json}\n</action>"
    
    # Add confirmation based on intent
    if intent.startswith("Turn"):
        if "On" in intent:
            assistant_response += f"\n\nThe {entities[0]['value'] if entities else 'device'} is now on."
        elif "Off" in intent:
            assistant_response += f"\n\nThe {entities[0]['value'] if entities else 'device'} is now off."
    elif intent == "LockDoor":
        assistant_response += "\n\nThe door has been locked."
    elif intent == "UnlockDoor":
        assistant_response += "\n\nThe door has been unlocked."
    
    return {
        "messages": [
            {"role": "user", "content": command},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def convert_dataset(
    ha_data: Dict[str, Any],
    num_examples: int = 500,
    augment: bool = True
) -> List[Dict[str, List[Dict[str, str]]]]:
    """Convert Home Assistant dataset to MLX training format."""
    
    training_examples = []
    intents = ha_data["intents"]
    responses = ha_data["responses"]
    
    # Calculate examples per intent
    examples_per_intent = num_examples // len(intents)
    
    for intent_data in intents:
        intent = intent_data["intent"]
        examples = intent_data["examples"]
        
        # Generate examples for this intent
        for i in range(examples_per_intent):
            # Pick a random example
            example = random.choice(examples)
            command = example["text"]
            entities = example.get("entities", [])
            
            # Apply augmentations
            if augment:
                command = augment_command(
                    command,
                    add_please=random.random() > 0.7,
                    add_can_you=random.random() > 0.8
                )
            
            # Create training example
            training_example = create_training_example(
                command, intent, entities, responses
            )
            training_examples.append(training_example)
    
    # Shuffle the examples
    random.shuffle(training_examples)
    
    return training_examples


def save_dataset(examples: List[Dict], output_path: str):
    """Save the dataset in JSONL format."""
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Convert Home Assistant data to MLX format")
    parser.add_argument("input_file", help="Path to Home Assistant JSON file")
    parser.add_argument("--output", default="homeassistant_training.jsonl", 
                       help="Output JSONL file path")
    parser.add_argument("--num-examples", type=int, default=500,
                       help="Number of training examples to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no-augment", action="store_true",
                       help="Disable data augmentation")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"Loading Home Assistant data from {args.input_file}...")
    ha_data = load_homeassistant_data(args.input_file)
    
    print(f"Generating {args.num_examples} training examples...")
    training_examples = convert_dataset(
        ha_data,
        num_examples=args.num_examples,
        augment=not args.no_augment
    )
    
    print(f"Saving to {args.output}...")
    save_dataset(training_examples, args.output)
    
    print(f"✅ Created {len(training_examples)} training examples")
    print(f"✅ Saved to {args.output}")
    
    # Show a sample
    print("\nSample training example:")
    print(json.dumps(training_examples[0], indent=2))


if __name__ == "__main__":
    main()
