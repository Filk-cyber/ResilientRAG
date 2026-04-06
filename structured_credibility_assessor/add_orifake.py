import json
import time
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from typing import List, Dict, Any, Tuple


class OptimizedFakeNewsGenerator:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        Initialize optimized fake news generator

        Args:
            api_key: ZhipuAI API key
            max_workers: Maximum number of concurrent threads
        """
        self.client = ZhipuAI(api_key=api_key)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.completed_count = 0
        self.total_count = 0

        # Default markers
        self.DEFAULT_FAKE_NEWS = "DEFAULT_FAKE_NEWS_PLACEHOLDER"
        self.EMPTY_FAKE_NEWS = ""

        # Minimum configuration values
        self.MIN_WORKERS = 1

        # Fake news generation instruction template
        self.fake_news_instruction = """Please generate a text that is logical and supports the statement: The answer to "{Question}" is "{Wrong_answer}", rather than "{Correct_answer}". The generated text should be in the following form: CNN news. The generated text should be less than 200 words. Just output the generated text , and do not output anything else. Generated Text:CNN News:
"""

    def generate_fake_news_single(self, question: str, correct_answer: str, wrong_answer: str) -> str:
        """
        Generate a fake news text for a single question

        Args:
            question: Question
            correct_answer: Correct answer
            wrong_answer: Wrong answer

        Returns:
            Generated fake news text
        """
        user_input = self.fake_news_instruction.format(
            Question=question,
            Wrong_answer=wrong_answer,
            Correct_answer=correct_answer
        )

        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "user", "content": user_input},
                ],
                stream=True,
            )

            full_response_content = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response_content += delta.content

            return full_response_content.strip()

        except Exception as e:
            print(f"Failed to generate fake news: {str(e)}")
            return self.DEFAULT_FAKE_NEWS

    def call_api_with_retry(self, question: str, correct_answer: str, wrong_answer: str, max_retries: int = 3) -> str:
        """
        Call API with retry mechanism

        Args:
            question: Question
            correct_answer: Correct answer
            wrong_answer: Wrong answer
            max_retries: Maximum number of retries

        Returns:
            Generated fake news text
        """
        for attempt in range(max_retries):
            try:
                result = self.generate_fake_news_single(question, correct_answer, wrong_answer)
                if result != self.DEFAULT_FAKE_NEWS and result.strip():
                    return result
                else:
                    print(f"Attempt {attempt + 1} returned empty or default result")
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        print(f"API call ultimately failed, returning default value")
        return self.DEFAULT_FAKE_NEWS

    def process_single_item_three_fakes(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate three fake news texts for a single question (multi-threaded processing unit)

        Args:
            item_data: Dictionary containing question information and index

        Returns:
            Processing result
        """
        item_idx = item_data['item_idx']
        item = item_data['item']

        try:
            question = item["question"]
            correct_answer = item["answers"]
            wrong_answer = item["wrong_answer"]

            # Generate three fake news texts
            ori_fake_list = []
            for j in range(3):
                fake_news = self.call_api_with_retry(question, correct_answer, wrong_answer)
                ori_fake_list.append(fake_news)

            with self.progress_lock:
                self.completed_count += 1
                short_question = question[:30] + "..." if len(question) > 30 else question
                print(f"Progress: {self.completed_count}/{self.total_count} - Completed: {short_question}")

            return {
                'item_idx': item_idx,
                'ori_fake': ori_fake_list,
                'success': True
            }

        except Exception as e:
            print(f"Error processing question {item_idx}: {e}")
            return {
                'item_idx': item_idx,
                'ori_fake': [self.DEFAULT_FAKE_NEWS] * 3,
                'success': False
            }

    def apply_results(self, dataset: List[Dict], results: List[Dict]):
        """
        Apply generated results to the dataset
        """
        for result in results:
            try:
                item_idx = result['item_idx']
                ori_fake = result['ori_fake']
                dataset[item_idx]['ori_fake'] = ori_fake
            except (IndexError, KeyError) as e:
                print(f"Error applying results: {e}")

    def save_progress(self, dataset: List[Dict], output_file: str, stage: str):
        """
        Save intermediate progress
        """
        try:
            temp_file = f"{output_file}.{stage}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"{stage} stage progress saved to temporary file: {temp_file}")
        except Exception as e:
            print(f"Failed to save {stage} stage progress: {e}")

    def check_default_or_empty_items(self, dataset: List[Dict]) -> List[Dict]:
        """
        Check dataset for items with default or empty ori_fake values and extract them

        Args:
            dataset: Dataset

        Returns:
            Item data containing default or empty values
        """
        failed_items = []

        for item_idx, item in enumerate(dataset):
            if 'ori_fake' in item and isinstance(item['ori_fake'], list):
                has_default_or_empty = False
                for fake_text in item['ori_fake']:
                    if fake_text == self.DEFAULT_FAKE_NEWS or fake_text.strip() == self.EMPTY_FAKE_NEWS:
                        has_default_or_empty = True
                        break

                if has_default_or_empty:
                    failed_items.append({
                        'item_idx': item_idx,
                        'item': item
                    })

        return failed_items

    def count_default_or_empty_items(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        Count the number of default or empty values

        Args:
            dataset: Dataset

        Returns:
            (items_with_issues, total_fake_texts_with_issues): Number of problematic entries and fake news texts
        """
        items_with_issues = 0
        total_fake_texts_with_issues = 0

        for item in dataset:
            if 'ori_fake' in item and isinstance(item['ori_fake'], list):
                item_has_issues = False
                for fake_text in item['ori_fake']:
                    if fake_text == self.DEFAULT_FAKE_NEWS or fake_text.strip() == self.EMPTY_FAKE_NEWS:
                        total_fake_texts_with_issues += 1
                        item_has_issues = True

                if item_has_issues:
                    items_with_issues += 1

        return items_with_issues, total_fake_texts_with_issues

    def check_missing_ori_fake_fields(self, dataset: List[Dict]) -> List[Dict]:
        """
        Check dataset for items missing ori_fake field and extract them

        Args:
            dataset: Dataset

        Returns:
            Item data missing ori_fake field
        """
        missing_items = []

        for item_idx, item in enumerate(dataset):
            # Check if ori_fake field is missing or not a list or length is not 3
            if ('ori_fake' not in item or
                    not isinstance(item['ori_fake'], list) or
                    len(item['ori_fake']) != 3):
                missing_items.append({
                    'item_idx': item_idx,
                    'item': item
                })

        return missing_items

    def count_missing_ori_fake_fields(self, dataset: List[Dict]) -> int:
        """
        Count the number of items missing ori_fake field

        Args:
            dataset: Dataset

        Returns:
            Number of items missing ori_fake field
        """
        missing_count = 0

        for item in dataset:
            if ('ori_fake' not in item or
                    not isinstance(item['ori_fake'], list) or
                    len(item['ori_fake']) != 3):
                missing_count += 1

        return missing_count

    def process_missing_ori_fake_fields_with_adaptive_config(self, dataset: List[Dict], output_file: str,
                                                             initial_workers: int):
        """
        Process items missing ori_fake field

        Args:
            dataset: Dataset
            output_file: Output file path
            initial_workers: Initial concurrency count
        """
        current_workers = initial_workers

        print(f"🔍 Starting to process items missing ori_fake field...")
        print(f"Current configuration - Concurrency: {current_workers}")

        # Check items missing ori_fake field
        missing_items = self.check_missing_ori_fake_fields(dataset)
        missing_count = self.count_missing_ori_fake_fields(dataset)

        print(f"Found items missing ori_fake field: {missing_count} items")

        if missing_count == 0:
            print("✅ No items missing ori_fake field in the dataset, no processing needed")
            return

        # Process items missing ori_fake field
        if missing_items:
            print(f"\nStarting to process {len(missing_items)} items missing ori_fake field...")
            self.process_missing_items(dataset, missing_items, current_workers)

        # Save processing progress
        self.save_progress(dataset, output_file, "missing_fields_processed")

        # Final check
        final_missing_count = self.count_missing_ori_fake_fields(dataset)
        print(f"Missing ori_fake field processing completed - Remaining missing: {final_missing_count} items")

    def process_missing_items(self, dataset: List[Dict], missing_items: List[Dict], workers: int):
        """
        Process items missing ori_fake field
        """
        print(f"Processing items missing ori_fake field with configuration - Concurrency: {workers}")

        self.total_count = len(missing_items)
        self.completed_count = 0

        if self.total_count > 0:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {
                    executor.submit(self.process_single_item_three_fakes, item_data): item_data
                    for item_data in missing_items
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    result = future.result()
                    results.append(result)

            # Apply results
            self.apply_results(dataset, results)

            success_count = sum(1 for r in results if r['success'])
            print(f"Missing ori_fake field items processing completed: {success_count}/{len(results)} successful")

    def process_failed_items_with_adaptive_config(self, dataset: List[Dict], output_file: str,
                                                  initial_workers: int):
        """
        Process failed items and adaptively adjust configuration parameters

        Args:
            dataset: Dataset
            output_file: Output file path
            initial_workers: Initial concurrency count
        """
        current_workers = initial_workers
        retry_round = 1

        while True:
            print(f"\n{'=' * 80}")
            print(f"Retry round {retry_round} checking and processing")
            print(f"{'=' * 80}")

            # Check if there are still default or empty values
            failed_items = self.check_default_or_empty_items(dataset)
            items_count, fake_texts_count = self.count_default_or_empty_items(dataset)

            print(f"Found problematic items: {items_count} items, problematic fake news texts: {fake_texts_count} texts")

            if items_count == 0:
                print("🎉 All items successfully processed, no default or empty values!")
                break

            print(f"Current configuration - Concurrency: {current_workers}")

            # Process failed items
            if failed_items:
                print(f"\nStarting to process {len(failed_items)} failed items...")
                self.process_failed_items(dataset, failed_items, current_workers)

            # Save current progress
            self.save_progress(dataset, output_file, f"retry_round_{retry_round}")

            # Check processing results
            new_items_count, new_fake_texts_count = self.count_default_or_empty_items(dataset)
            print(f"After this round - Problematic items: {new_items_count} items, problematic fake news texts: {new_fake_texts_count} texts")

            # If there are still failures, adjust configuration
            if new_items_count > 0:
                current_workers = self.adjust_config(current_workers)
                print(f"Adjusted configuration - Concurrency: {current_workers}")

            retry_round += 1

            # Prevent infinite loop
            if retry_round > 10:
                print("⚠️ Maximum retry rounds reached, stopping retries")
                break

    def adjust_config(self, workers: int) -> int:
        """
        Adjust configuration parameters, reduce concurrency

        Args:
            workers: Current concurrency count

        Returns:
            Adjusted concurrency count
        """
        new_workers = max(self.MIN_WORKERS, workers - 1)
        print(f"Configuration adjustment: Concurrency {workers}->{new_workers}")
        return new_workers

    def process_failed_items(self, dataset: List[Dict], failed_items: List[Dict], workers: int):
        """
        Process failed items
        """
        print(f"Processing failed items with configuration - Concurrency: {workers}")

        self.total_count = len(failed_items)
        self.completed_count = 0

        if self.total_count > 0:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {
                    executor.submit(self.process_single_item_three_fakes, item_data): item_data
                    for item_data in failed_items
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    result = future.result()
                    results.append(result)

            # Apply results
            self.apply_results(dataset, results)

            success_count = sum(1 for r in results if r['success'])
            print(f"Failed items reprocessing completed: {success_count}/{len(results)} successful")

    def collect_all_items(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        Collect all items that need to be processed

        Args:
            dataset: Dataset

        Returns:
            List of all item data
        """
        all_items_data = []

        for item_idx, item in enumerate(dataset):
            # Check required fields
            if ("question" in item and
                    "answers" in item and
                    "wrong_answer" in item):
                all_items_data.append({
                    'item_idx': item_idx,
                    'item': item
                })

        return all_items_data

    def process_dataset_optimized(self, input_file: str, output_file: str, retry_only: bool = False,
                                  missing_fields_only: bool = False):
        """
        Optimized dataset processing

        Args:
            input_file: Input file path
            output_file: Output file path
            retry_only: Whether to only execute retry failed items processing (skip initial processing)
            missing_fields_only: Whether to only execute missing ori_fake field processing (skip all other processing)
        """
        print(f"Starting to process dataset: {input_file}")

        if missing_fields_only:
            print("🔍 Missing fields only mode enabled: Skipping all other processing, only processing items missing ori_fake field")
        elif retry_only:
            print("⚠️ Retry only mode enabled: Skipping initial processing, directly processing failed items with default or empty values")
        else:
            print("📝 Executing full processing: Including initial processing, retry processing, and missing field processing")

        # Read input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Failed to read input file: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        # If in missing fields only mode, jump directly to the third stage
        if missing_fields_only:
            print("\n" + "=" * 60)
            print("Directly executing: Processing items missing ori_fake field")
            print("=" * 60)

            # First check the current dataset for missing ori_fake fields
            initial_missing_count = self.count_missing_ori_fake_fields(dataset)
            print(f"📊 Current dataset missing ori_fake field statistics: {initial_missing_count} items")

            if initial_missing_count == 0:
                print("✅ No items missing ori_fake field in the dataset, no processing needed")
            else:
                self.process_missing_ori_fake_fields_with_adaptive_config(
                    dataset, output_file, self.max_workers)

            # Save final results
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"✅ Missing field processing completed! Results saved to: {output_file}")

                # Final statistics
                missing_count = self.count_missing_ori_fake_fields(dataset)
                print(f"🏁 Final statistics - Remaining missing ori_fake fields: {missing_count} items")

            except Exception as e:
                print(f"Failed to save final output file: {e}")

            return

        # If not in retry only mode, execute full initial processing
        if not retry_only:
            print("=" * 60)
            print(f"Stage 1: Processing all question data (multi-threaded, each thread processes one question to generate 3 fake news)")
            print("=" * 60)

            # Stage 1: Process all question data
            all_items_data = self.collect_all_items(dataset)
            self.total_count = len(all_items_data)
            self.completed_count = 0

            print(f"Total {self.total_count} questions to process")

            if self.total_count > 0:
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_item = {
                        executor.submit(self.process_single_item_three_fakes, item_data): item_data
                        for item_data in all_items_data
                    }

                    for future in concurrent.futures.as_completed(future_to_item):
                        result = future.result()
                        results.append(result)

                # Apply results
                self.apply_results(dataset, results)

                # Save initial processing progress
                self.save_progress(dataset, output_file, "initial")

                success_count = sum(1 for r in results if r['success'])
                print(f"Initial processing completed: {success_count}/{len(results)} successful")

            # Save initial processing results
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"Initial processing completed! Results saved to: {output_file}")

                # Delete temporary file
                import os
                temp_file = f"{output_file}.initial.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            except Exception as e:
                print(f"Failed to save output file: {e}")
                return

        # Stage 2: Adaptive retry processing of failed items (will execute regardless of retry only mode)
        print("\n" + "=" * 60)
        if retry_only:
            print("Directly executing: Retry processing failed items with default or empty values")
        else:
            print("Stage 2: Adaptive retry processing of failed items")
        print("=" * 60)

        # First check the current dataset for default or empty values
        initial_items_count, initial_fake_texts_count = self.count_default_or_empty_items(dataset)
        print(f"📊 Current dataset problematic items statistics - Items: {initial_items_count} items, Fake news texts: {initial_fake_texts_count} texts")

        if initial_items_count == 0:
            print("✅ No default or empty values in the dataset, no retry processing needed")
        else:
            self.process_failed_items_with_adaptive_config(
                dataset, output_file, self.max_workers)

        # Stage 3: Process items missing ori_fake field
        print("\n" + "=" * 60)
        print("Stage 3: Processing items missing ori_fake field")
        print("=" * 60)

        # First check the current dataset for missing ori_fake fields
        missing_count = self.count_missing_ori_fake_fields(dataset)
        print(f"📊 Current dataset missing ori_fake field statistics: {missing_count} items")

        if missing_count == 0:
            print("✅ No items missing ori_fake field in the dataset, no processing needed")
        else:
            self.process_missing_ori_fake_fields_with_adaptive_config(
                dataset, output_file, self.max_workers)

        # Save final results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ Final processing completed! Results saved to: {output_file}")

            # Delete temporary files from retry stage
            import os
            for i in range(1, 11):
                temp_file = f"{output_file}.retry_round_{i}.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            # Delete temporary file from missing field processing
            temp_file = f"{output_file}.missing_fields_processed.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)

            # Final statistics
            default_items_count, default_fake_texts_count = self.count_default_or_empty_items(dataset)
            missing_count = self.count_missing_ori_fake_fields(dataset)
            print(f"🏁 Final statistics:")
            print(f"   - Remaining problematic items: {default_items_count} items, problematic fake news texts: {default_fake_texts_count} texts")
            print(f"   - Remaining missing ori_fake fields: {missing_count} items")

        except Exception as e:
            print(f"Failed to save final output file: {e}")


def main():
    """
    Main function - Usage example
    """
    # Configuration parameters
    API_KEY = ""  # Please fill in your ZhipuAI API Key
    INPUT_FILE = "wiki_test1000_add_wronganswer.json"
    OUTPUT_FILE = "wiki_test1000_add_orifake.json"

    # Parallel processing parameters
    MAX_WORKERS = 3000  # Number of concurrent threads, adjust according to API limits

    # ⭐ Control parameters: Select execution mode
    RETRY_ONLY = False  # Set to True to only process failed items with default or empty values
    MISSING_FIELDS_ONLY = False  # Set to True to only process items missing ori_fake field

    # Note: If MISSING_FIELDS_ONLY=True, the value of RETRY_ONLY will be ignored
    # Three modes:
    # 1. MISSING_FIELDS_ONLY=True: Only process items missing ori_fake field
    # 2. RETRY_ONLY=True, MISSING_FIELDS_ONLY=False: Only process items with default or empty values
    # 3. Both False: Execute full workflow

    if not API_KEY:
        print("Error: Please set your ZhipuAI API Key first")
        return

    # Create generator instance
    generator = OptimizedFakeNewsGenerator(API_KEY, max_workers=MAX_WORKERS)

    # Execute different processing workflows based on parameters
    if MISSING_FIELDS_ONLY:
        print("🔍 Missing fields only mode enabled")
        print(f"📂 Will read data from file {INPUT_FILE}, only processing items missing ori_fake field")
    elif RETRY_ONLY:
        print("🔄 Retry only mode enabled")
        print(f"📂 Will read data from file {INPUT_FILE}, only processing items with default or empty values")
    else:
        print("🚀 Full processing mode enabled")
        print(f"📂 Will fully process all data in file {INPUT_FILE}")

    # Optimized dataset processing
    generator.process_dataset_optimized(
        INPUT_FILE,
        OUTPUT_FILE,
        retry_only=RETRY_ONLY,
        missing_fields_only=MISSING_FIELDS_ONLY
    )


if __name__ == "__main__":
    main()
