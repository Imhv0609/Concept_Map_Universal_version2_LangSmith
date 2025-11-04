"""
Universal Concept Map Teaching Agent - Main Application

This system can analyze ANY topic and create comprehensive concept maps
with subtopics, hierarchies, and educational metadata - not limited to predefined JSON files.
"""

import os
import json
import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from graph import create_description_based_concept_map_graph, print_description_based_workflow_summary
from states import ConceptMapState
from description_analyzer import extract_topic_name_from_description
from graph_visualizer import ConceptMapVisualizer
from token_tracker import get_tracker, reset_tracker

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check LangSmith configuration
LANGSMITH_ENABLED = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'
if LANGSMITH_ENABLED:
    logger.info("🔍 LangSmith tracing ENABLED - Performance metrics will be tracked")
    logger.info(f"📊 LangSmith Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    logger.info(f"🌐 View traces at: https://smith.langchain.com")
else:
    logger.info("ℹ️  LangSmith tracing disabled - Set LANGCHAIN_TRACING_V2=true in .env to enable")


def get_educational_levels():
    """Get list of supported educational levels"""
    return [
        "elementary",
        "middle school", 
        "high school",
        "undergraduate",
        "graduate",
        "professional",
        "general audience"
    ]


def run_description_based_concept_mapping(description, educational_level="high school", topic_name=None, tts_enabled=True):
    """
    Run the description-based concept mapping workflow
    
    Args:
        description (str): The description text to analyze (1 word to 3000+ words) - PRIMARY INPUT
        educational_level (str): Target educational level (default: "high school") 
        topic_name (str): Optional topic name (auto-extracted if None)
        tts_enabled (bool): Enable text-to-speech narration (default: True - always enabled)
    """
    
    print("🚀 Description-Based LLM-Powered Concept Map Teaching Agent")
    print("=" * 70)
    print_description_based_workflow_summary()
    
    # Text-to-Speech narration (if enabled)
    if tts_enabled:
        try:
            from tts_handler import TTSHandler
            logger.info("🎤 Text-to-Speech enabled - Starting narration...")
            tts = TTSHandler(rate=190, volume=0.9)  # Faster speech rate
            tts.speak_text_sentence_by_sentence(description, pause_duration=1.0)
            logger.info("✅ Narration complete - Proceeding to concept map generation...")
        except ImportError:
            logger.warning("⚠️  pyttsx3 not installed. Install with: pip install pyttsx3")
            logger.info("Continuing without text-to-speech...")
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            logger.info("Continuing without text-to-speech...")
    
    # Auto-extract topic name if not provided
    if not topic_name:
        topic_name = extract_topic_name_from_description(description)
    
    # Initialize state for new description-based approach
    initial_state = ConceptMapState(
        description=description,
        educational_level=educational_level,
        topic_name=topic_name,
        description_analysis={},
        complexity_config={},
        extracted_concepts=[],
        concept_relationships=[],
        concept_hierarchy=[],
        enriched_concepts={},
        learning_objectives=[],
        teaching_strategies=[],
        # Legacy fields for backward compatibility
        raw_subtopics=[],
        subtopic_concepts={},
        key_concepts_per_subtopic={},
        subtopic_hierarchies={},
        cross_subtopic_links=[],
        enriched_subtopics={},
        processing_log=[],
        errors=[],
        success=True,
        timestamp=datetime.now().isoformat()
    )
    
    try:
        # Reset token tracker for this run
        reset_tracker()
        
        # Create and run the new workflow
        logger.info("🔄 Starting description-based concept mapping workflow...")
        
        workflow = create_description_based_concept_map_graph()
        final_state = workflow.invoke(initial_state)
        
        # Log token usage summary
        get_tracker().log_summary()
        
        # Print results
        print_results(final_state)
        
        # Save results and get the filepath
        json_filepath = save_results(final_state)
        
        # Store the JSON filepath in the result for graph generation
        final_state['_json_filepath'] = json_filepath
        
        return final_state
        
    except Exception as e:
        logger.error(f"❌ Error in workflow: {e}")
        return None


# Legacy function for backward compatibility
def run_universal_concept_mapping(topic_name, educational_level="high school", topic_description=""):
    """
    Legacy function - converts to new description-based approach
    
    Args:
        topic_name (str): The main topic/chapter to analyze (required)
        educational_level (str): Target educational level (default: "high school")
        topic_description (str): Additional context about the topic
    """
    
    # Convert old parameters to new description-based approach
    if topic_description:
        description = f"{topic_name}: {topic_description}"
    else:
        description = topic_name
    
    return run_description_based_concept_mapping(
        description=description,
        educational_level=educational_level,
        topic_name=topic_name
    )


def print_results(state: ConceptMapState):
    """
    Print the results of the description-based concept mapping workflow
    """
    print("\n" + "=" * 70)
    print("📊 DESCRIPTION-BASED CONCEPT MAP RESULTS")
    print("=" * 70)
    
    print(f"📚 Topic: {state['topic_name']}")
    print(f"🎓 Educational Level: {state['educational_level']}")
    print(f"📝 Description: {state['description'][:200]}{'...' if len(state['description']) > 200 else ''}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Description analysis statistics
    if state.get('description_analysis'):
        analysis = state['description_analysis']
        print("📈 Description Analysis:")
        print(f"  • Word count: {analysis.get('word_count', 0)}")
        print(f"  • Unique words: {analysis.get('unique_words', 0)}")
        print(f"  • Sentences: {analysis.get('sentence_count', 0)}")
        print(f"  • Complexity level: {analysis.get('complexity', {}).get('detail_level', 'Unknown')}")
        print()
    
    # Concept extraction statistics
    extracted_concepts = state.get('extracted_concepts', [])
    concept_relationships = state.get('concept_relationships', [])
    concept_hierarchy = state.get('concept_hierarchy', [])
    
    print("📈 Concept Map Statistics:")
    print(f"  • Extracted concepts: {len(extracted_concepts)}")
    print(f"  • Concept relationships: {len(concept_relationships)}")
    print(f"  • Hierarchy levels: {len(concept_hierarchy)}")
    print(f"  • Enriched concepts: {len(state.get('enriched_concepts', {}))}")
    print()
    
    # Extracted concepts overview
    if extracted_concepts:
        print("🎯 Extracted Concepts:")
        for i, concept in enumerate(extracted_concepts, 1):
            concept_type = concept.get('type', 'Unknown')
            importance = concept.get('importance', 'Unknown')
            print(f"  {i}. {concept['name']} ({concept_type}, {importance})")
            if concept.get('definition'):
                definition = concept['definition'][:100] + "..." if len(concept['definition']) > 100 else concept['definition']
                print(f"     └─ {definition}")
        print()
    
    # Concept relationships overview
    if concept_relationships:
        print("🔗 Key Concept Relationships:")
        for i, rel in enumerate(concept_relationships[:10], 1):  # Show first 10 relationships
            from_concept = rel.get('from_concept', 'Unknown')
            to_concept = rel.get('to_concept', 'Unknown')
            rel_type = rel.get('relationship_type', 'related_to')
            strength = rel.get('strength', 'Unknown')
            print(f"  {i}. {from_concept} → {rel_type} → {to_concept} ({strength})")
            if rel.get('relationship_description'):
                desc = rel['relationship_description'][:80] + "..." if len(rel['relationship_description']) > 80 else rel['relationship_description']
                print(f"     └─ {desc}")
        if len(concept_relationships) > 10:
            print(f"     ... and {len(concept_relationships) - 10} more relationships")
        print()
    
    # Learning hierarchy overview
    if concept_hierarchy:
        print("🏗️ Learning Hierarchy:")
        for level in concept_hierarchy:
            level_name = level.get('level_name', f"Level {level.get('level', 'Unknown')}")
            concepts = level.get('concepts', [])
            difficulty = level.get('difficulty', 'Unknown')
            print(f"  📚 {level_name} ({difficulty})")
            if level.get('level_description'):
                print(f"     {level['level_description']}")
            for concept in concepts[:5]:  # Show first 5 concepts per level
                concept_name = concept.get('name', 'Unknown') if isinstance(concept, dict) else concept
                print(f"     • {concept_name}")
            if len(concepts) > 5:
                print(f"     ... and {len(concepts) - 5} more concepts")
            print()
    
    # Learning objectives
    learning_objectives = state.get('learning_objectives', [])
    if learning_objectives:
        print("🎯 Learning Objectives:")
        for i, objective in enumerate(learning_objectives[:5], 1):  # Show first 5 objectives
            print(f"  {i}. {objective}")
        if len(learning_objectives) > 5:
            print(f"     ... and {len(learning_objectives) - 5} more objectives")
        print()
    
    # Processing log
    if state.get('processing_log'):
        print("📋 Processing Summary:")
        for log_entry in state['processing_log']:
            print(f"  {log_entry}")
        print()
    
    # Error reporting
    if state.get('errors'):
        print("⚠️  Errors Encountered:")
        for error in state['errors']:
            print(f"  • {error}")
        print()
    
    success_status = "✅ Success" if state.get('success', False) else "❌ Failed"
    print(f"Status: {success_status}")
    print("=" * 70)


def save_results(state: ConceptMapState):
    """
    Save the concept map results to JSON file
    
    Returns:
        str: Path to the saved JSON file, or None if failed
    """
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_name_clean = state['topic_name'].replace(' ', '_').replace('/', '_')
    filename = f"description_based_concept_map_{topic_name_clean}_{timestamp}.json"
    filepath = output_dir / filename
    
    # Prepare data for saving
    output_data = {
        "metadata": {
            "topic_name": state['topic_name'],
            "educational_level": state['educational_level'],
            "description": state['description'],
            "timestamp": state['timestamp'],
            "processing_date": datetime.now().isoformat(),
            "success": state['success']
        },
        "description_analysis": state.get('description_analysis', {}),
        "complexity_config": state.get('complexity_config', {}),
        "extracted_concepts": state.get('extracted_concepts', []),
        "concept_relationships": state.get('concept_relationships', []),
        "concept_hierarchy": state.get('concept_hierarchy', []),
        "enriched_concepts": state.get('enriched_concepts', {}),
        "learning_objectives": state.get('learning_objectives', []),
        "teaching_strategies": state.get('teaching_strategies', []),
        "processing_log": state.get('processing_log', []),
        "errors": state.get('errors', [])
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
        logger.info(f"Results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        error_msg = f"Failed to save results: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return None


def interactive_mode():
    """
    Interactive mode for description-based concept mapping
    """
    print("🚀 Interactive Description-Based Concept Map Generator")
    print("=" * 60)
    print("Enter a description of any length (1 word to 3000+ words)")
    print("The system will automatically scale complexity to match your description.")
    print()
    
    while True:
        try:
            # Get description input
            print("📝 Enter your description (or 'quit' to exit):")
            description = input("> ").strip()
            
            if description.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not description:
                print("⚠️  Please enter a description.")
                continue
            
            # Get educational level
            print("\n🎓 Select educational level:")
            levels = get_educational_levels()
            for i, level in enumerate(levels, 1):
                print(f"  {i}. {level}")
            
            while True:
                try:
                    level_choice = input("\nEnter number (or press Enter for 'high school'): ").strip()
                    if not level_choice:
                        educational_level = "high school"
                        break
                    else:
                        level_idx = int(level_choice) - 1
                        if 0 <= level_idx < len(levels):
                            educational_level = levels[level_idx]
                            break
                        else:
                            print("⚠️  Invalid choice. Please try again.")
                except ValueError:
                    print("⚠️  Please enter a valid number.")
            
            # Optional topic name
            topic_name = input("\n📚 Topic name (optional, auto-extracted if blank): ").strip() or None
            
            # Ask if user wants dynamic or static mode
            print("\n🎨 Visualization mode:")
            print("  1. Dynamic (real-time Streamlit visualization) [DEFAULT]")
            print("  2. Static (JSON + PNG files)")
            mode_choice = input("\nEnter number (or press Enter for dynamic): ").strip()
            use_dynamic = True  # Default to dynamic
            if mode_choice == '2':
                use_dynamic = False
            
            print("\n" + "=" * 60)
            
            # Run dynamic or static mode based on choice
            if use_dynamic:
                # Run dynamic mode with Streamlit
                from dynamic_orchestrator import run_dynamic_mode
                
                # Extract topic name if not provided
                if not topic_name:
                    topic_name = extract_topic_name_from_description(description)
                
                logger.info("🚀 Interactive mode - launching dynamic visualization...")
                success = run_dynamic_mode(description, educational_level, topic_name)
                
                if success:
                    print("\n✅ Dynamic concept map session completed!")
                else:
                    print("\n❌ Dynamic concept map failed.")
                
                # Ask if user wants to continue
                continue_choice = input("\n🔄 Generate another concept map? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("👋 Goodbye!")
                    break
            else:
                # Run static mode (original behavior)
                result = run_description_based_concept_mapping(
                    description=description,
                    educational_level=educational_level,
                    topic_name=topic_name
                )
                
                if result and result.get('success'):
                    print("\n✅ Concept mapping completed successfully!")
                    
                    # Generate visualization automatically
                    print("\n🎨 Generating concept map visualization...")
                    try:
                        json_filepath = result.get('_json_filepath')
                        if json_filepath and Path(json_filepath).exists():
                            visualizer = ConceptMapVisualizer()
                            logger.info(f"Loading visualization from: {json_filepath}")
                            if visualizer.load_from_json(json_filepath):
                                stats = visualizer.extract_graph_data()
                                logger.info(f"Graph stats: {stats}")
                                image_path = visualizer.generate_graph_image()
                                if image_path:
                                    print(f"📊 Concept map visualization saved: {image_path}")
                                else:
                                    print("❌ Failed to generate concept map visualization")
                            else:
                                print("❌ Failed to load JSON data for visualization")
                        else:
                            print("❌ JSON file not found for visualization")
                    except Exception as e:
                        print(f"❌ Error generating visualization: {e}")
                        logger.error(f"Visualization error: {e}")
                    
                    # Ask if user wants to continue
                    continue_choice = input("\n🔄 Generate another concept map? (y/n): ").strip().lower()
                    if continue_choice not in ['y', 'yes']:
                        print("👋 Goodbye!")
                        break
                else:
                    print("\n❌ Concept mapping failed. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


def main():
    """
    Main function with command line argument parsing
    """
    parser = argparse.ArgumentParser(
        description="Description-Based Universal Concept Map Teaching Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main_universal.py
  
  # Single word
  python main_universal.py --description "Photosynthesis" --level "elementary"
  
  # Short description  
  python main_universal.py --description "How plants make food using sunlight"
  
  # Long description
  python main_universal.py --description "Photosynthesis is a complex process..." --level "graduate"
  
  # With custom topic name
  python main_universal.py --description "Plants use sunlight to make energy" --topic "Plant Biology" --level "high school"
  
  # Generate both JSON and visual concept map
  python main_universal.py --description "Photosynthesis process" --generate-graph
  
  # Short form with graph generation
  python main_universal.py -d "Gravity and motion" -l "middle school" -g
"""
    )
    
    parser.add_argument(
        "--description", "-d",
        type=str,
        help="Description text to analyze (1 word to 3000+ words)"
    )
    
    parser.add_argument(
        "--level", "-l",
        type=str,
        default="high school",
        choices=get_educational_levels(),
        help="Educational level (default: high school)"
    )
    
    parser.add_argument(
        "--topic", "-t",
        type=str,
        help="Topic name (optional, auto-extracted if not provided)"
    )
    
    parser.add_argument(
        "--generate-graph", "-g",
        action="store_true",
        help="Generate visual concept map image along with JSON output"
    )
    
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable text-to-speech narration (reads description sentence-by-sentence)"
    )
    
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Enable dynamic concept map generation with real-time updates (launches Streamlit web interface) [DEFAULT]"
    )
    
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static mode (original behavior: JSON output only, no Streamlit visualization)"
    )
    
    # Legacy arguments for backward compatibility
    parser.add_argument(
        "--topic-name",
        type=str,
        help="[LEGACY] Topic name (use --description instead)"
    )
    
    parser.add_argument(
        "--topic-description",
        type=str,
        help="[LEGACY] Topic description (use --description instead)"
    )
    
    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.topic_name and not args.description:
        description = args.topic_name
        if args.topic_description:
            description = f"{args.topic_name}: {args.topic_description}"
        topic_name = args.topic_name
    elif args.description:
        description = args.description
        topic_name = args.topic
    else:
        # No arguments provided, run interactive mode
        interactive_mode()
        return
    
    # Route to dynamic mode by default (unless --static flag is set)
    if not args.static:
        logger.info("🚀 Dynamic mode enabled - launching Streamlit visualization...")
        from dynamic_orchestrator import run_dynamic_mode
        
        # Extract topic name if not provided
        if not topic_name:
            topic_name = extract_topic_name_from_description(description)
        
        success = run_dynamic_mode(description, args.level, topic_name)
        if success:
            print("\n✅ Dynamic concept map session completed!")
        else:
            print("\n❌ Dynamic concept map failed.")
            sys.exit(1)
        return
    
    # Run static description-based concept mapping (TTS always enabled)
    result = run_description_based_concept_mapping(
        description=description,
        educational_level=args.level,
        topic_name=topic_name,
        tts_enabled=True  # TTS always enabled by default
    )
    
    if result and result.get('success'):
        print("\n✅ Concept mapping completed successfully!")
        
        # Always generate graph in static mode
        print("\n🎨 Generating concept map visualization...")
        try:
            # Use the exact JSON file that was just created
            json_filepath = result.get('_json_filepath')
            if json_filepath and Path(json_filepath).exists():
                # Create visualizer and generate graph
                visualizer = ConceptMapVisualizer()
                logger.info(f"Loading visualization from: {json_filepath}")
                if visualizer.load_from_json(json_filepath):
                    stats = visualizer.extract_graph_data()
                    logger.info(f"Graph stats: {stats}")
                    image_path = visualizer.generate_graph_image()
                    if image_path:
                        print(f"📊 Concept map visualization saved: {image_path}")
                    else:
                        print("❌ Failed to generate concept map visualization")
                else:
                    print("❌ Failed to load JSON data for visualization")
            else:
                print("❌ JSON file not found for visualization")
        except Exception as e:
            print(f"❌ Error generating visualization: {e}")
            logger.error(f"Visualization error: {e}")
    else:
        print("\n❌ Concept mapping failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()