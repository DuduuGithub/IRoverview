import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from .api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param 
import tiktoken
from . import prompts as prompts_module
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_fixed



class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None, # For deterministic ouptputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
        ):
        if model is None:
            model = self.default_model
        
        # 创建基本参数
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }
        
        # Reasoning models do not support temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature
            
        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            # 使用OpenAI正确的结构化输出格式
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # 对于Pydantic模型，使用beta API的parse方法
                try:
                    params["response_format"] = response_format
                    completion = self.llm.beta.chat.completions.parse(**params)
                    response = completion.choices[0].message.parsed
                    content = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
                except Exception as e:
                    print(f"结构化输出错误: {e}")
                    # 回退到普通JSON响应格式
                    # 确保系统提示中包含"json"关键词
                    params["messages"][0]["content"] = system_content + " Please respond with a valid JSON object."
                    params["response_format"] = {"type": "json_object"}
                    completion = self.llm.chat.completions.create(**params)
                    content = completion.choices[0].message.content
                    try:
                        # 尝试解析JSON
                        import json
                        content = json.loads(content)
                    except:
                        pass
            else:
                # 普通的JSON响应格式
                # 确保系统提示中包含"json"关键词
                params["messages"][0]["content"] = system_content + " Please respond with a valid JSON object."
                params["response_format"] = {"type": "json_object"}
                completion = self.llm.chat.completions.create(**params)
                content = completion.choices[0].message.content
                try:
                    # 尝试解析JSON
                    import json
                    content = json.loads(content)
                except:
                    pass

        self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens}
        print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """Check the current balance for the provided token."""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """Get a list of available foundation models."""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """Get vector embeddings for the provided text inputs."""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts_module.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts_module.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

     
class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 20 seconds before retry...\n")

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Wrapper for generate_content with retry logic"""
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts_module.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts_module.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts_module.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini"] ="openai"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        self.response_data = None  # 初始化response_data属性

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        result = self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )
        # 获取底层处理器的response_data属性
        self.response_data = self.processor.response_data
        return result

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        """
        Get answers from RAG context
        
        Args:
            question: User question
            rag_context: RAG retrieval context
            schema: Answer schema ("name", "number", "boolean", "names", "comparative", "free_response" or "text")
            model: Model to use
            
        Returns:
            Generated answer as a dictionary
        """
        # Build appropriate prompts
        system_prompt, human_content, response_format = self._build_rag_context_prompts(schema)
        
        # Replace placeholders
        human_content = human_content.format(
            context=rag_context,
            question=question
        )
        
        # Determine whether to use structured output
        is_structured = schema != "text" and schema != "free_response"  # 修改判断条件，free_response也是非结构化的
        
        # Send request
        try:
            result = self.send_message(
                model=model,
                system_content=system_prompt,
                human_content=human_content,
                is_structured=is_structured,
                response_format=response_format
            )
            
            # 确保返回的是一个字典，包含所需的字段
            if schema == "text" or schema == "free_response":
                # 对于文本/自由响应，确保返回一个包含所需字段的字典
                if isinstance(result, str):
                    return {
                        "final_answer": result,
                        "step_by_step_analysis": "",
                        "reasoning_summary": "",
                        "relevant_pages": []
                    }
                else:
                    # 如果已经是字典但缺少一些字段，添加默认值
                    if not isinstance(result, dict):
                        result = {"final_answer": str(result)}
                    if "final_answer" not in result and "answer" in result:
                        result["final_answer"] = result["answer"]
                    if "step_by_step_analysis" not in result:
                        result["step_by_step_analysis"] = ""
                    if "reasoning_summary" not in result:
                        result["reasoning_summary"] = ""
                    if "relevant_pages" not in result:
                        result["relevant_pages"] = []
                    return result
            
            return result
        except Exception as e:
            import traceback
            print(f"Error while getting answer from RAG context: {str(e)}")
            print(traceback.format_exc())
            return {
                "step_by_step_analysis": f"Error generating answer: {str(e)}", 
                "reasoning_summary": "Error occurred", 
                "relevant_pages": [], 
                "final_answer": f"Error: {str(e)}"
            }

    def _build_rag_context_prompts(self, schema):
        """
        Build appropriate prompts for different question types
        
        Args:
            schema: Answer schema ("name", "number", "boolean", "names", "comparative", "free_response" or "text")
            
        Returns:
            system_prompt, human_content, response_format
        """
        # 使用已经导入的prompts_module而不是重新导入
        
        # Choose appropriate prompts based on schema
        if schema == "name":
            prompt_class = prompts_module.AnswerWithRAGContextNamePrompt
            response_format = prompt_class.AnswerSchema
        elif schema == "number":
            prompt_class = prompts_module.AnswerWithRAGContextNumberPrompt
            response_format = prompt_class.AnswerSchema
        elif schema == "boolean":
            prompt_class = prompts_module.AnswerWithRAGContextBooleanPrompt
            response_format = prompt_class.AnswerSchema
        elif schema == "names":
            prompt_class = prompts_module.AnswerWithRAGContextNamesPrompt
            response_format = prompt_class.AnswerSchema
        elif schema == "comparative":
            prompt_class = prompts_module.ComparativeAnswerPrompt
            response_format = prompt_class.AnswerSchema
        elif schema == "text" or schema == "free_response":  # 添加对free_response的支持
            # Text mode uses a simpler prompt
            system_prompt = """You are a RAG (Retrieval-Augmented Generation) answering system specialized in academic research.
Your task is to answer questions based on the provided academic papers or research document fragments.
Your answers should be based entirely on the provided context information, without adding external knowledge.
If the context doesn't contain sufficient information, clearly state that you cannot answer."""
            human_content = prompts_module.AnswerWithRAGContextSharedPrompt.user_prompt
            return system_prompt, human_content, None
        else:
            raise ValueError(f"Unsupported schema type: {schema}")
        
        # Use system prompt with schema
        system_prompt = prompt_class.system_prompt_with_schema
        human_content = prompt_class.user_prompt
        
        return system_prompt, human_content, response_format

    def get_rephrased_questions(self, original_question: str, documents: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        answer_dict = self.processor.send_message(
            system_content=prompts_module.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts_module.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                documents=", ".join([f'"{document}"' for document in documents])
            ),
            is_structured=True,
            response_format=prompts_module.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["document_id"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict


class AsyncOpenaiProcessor:
    
    def _get_unique_filepath(self, base_filepath):
        """Helper method to get unique filepath"""
        if not os.path.exists(base_filepath):
            return base_filepath
        
        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None
    ):
        # Create requests for jsonl
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            jsonl_requests.append(request)
            
        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""

                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer
                })
            
            # Sort by original index and build final list
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer']} 
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]
            
            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])
            
            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")
            
        return validated_data_list

    def rerank(self, query, documents):
        """
        重新排序文档列表
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            重新排序后的文档列表
        """
        # 保留原始方法以保持向后兼容性
        return self.rerank_documents(query, documents)
        
    def rerank_documents(self, query, documents):
        """
        重新排序文档列表
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            重新排序后的文档列表
        """
