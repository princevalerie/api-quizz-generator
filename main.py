from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
import json
import re
from datetime import datetime
import uvicorn
import asyncio
import concurrent.futures
from functools import wraps

app = FastAPI(
    title="AI Tutor Quiz Generator API",
    description="A FastAPI-based quiz generation and evaluation system powered by Google Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuizRequest(BaseModel):
    topic: str
    difficulty: str
    num_questions: int
    use_search: Optional[bool] = False  # Make search optional

class QuizResponse(BaseModel):
    quiz_id: str
    topic: str
    difficulty: str
    questions: List[Dict[str, Any]]
    created_at: str

class AnswerRequest(BaseModel):
    quiz_id: str
    question_id: int
    user_answer: str

class EvaluationResponse(BaseModel):
    is_correct: bool
    score: int
    feedback: str
    similarity_score: Optional[int] = None

class QuizEvaluationRequest(BaseModel):
    quiz_id: str
    answers: Dict[int, str]  # question_id -> user_answer

class QuizResultsResponse(BaseModel):
    quiz_id: str
    total_score: int
    max_score: int
    percentage: float
    grade: str
    correct_count: int
    total_questions: int
    evaluations: Dict[int, EvaluationResponse]

# In-memory storage for quizzes
quizzes = {}

def timeout_handler(timeout_seconds):
    """Decorator to handle function timeouts"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    return None
        return wrapper
    return decorator

class AITutor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        try:
            self.search = DuckDuckGoSearchRun()
        except Exception as e:
            print(f"Warning: Could not initialize search: {e}")
            self.search = None
    
    @timeout_handler(1)  # 10 second timeout
    def _search_with_timeout(self, query: str) -> str:
        """Protected search method with timeout"""
        if not self.search:
            return ""
        return self.search.run(query)
        
    def search_topic_info(self, topic: str, use_search: bool = False) -> str:
        """Fetch the latest information about a topic from the internet"""
        if not use_search or not self.search:
            return ""
            
        try:
            query = f"{topic} tutorial examples latest 2025"
            results = self._search_with_timeout(query)
            return results[:2000] if results else ""
        except Exception as e:
            print(f"Search warning: {str(e)}")
            return ""  # Don't fail, just return empty string
    
    def generate_quiz(self, topic: str, difficulty: str, num_questions: int, use_search: bool = False) -> Dict:
        """Generate a quiz based on a topic with varied question types"""
        
        # Try to fetch latest topic info, but don't fail if it doesn't work
        search_results = self.search_topic_info(topic, use_search)
        search_context = f"\nLatest information about the topic:\n{search_results}\n" if search_results else ""
        
        # Ensure we have at least 1 question and reasonable distribution
        num_questions = max(1, num_questions)
        
        # Calculate number of questions for each type
        if num_questions == 1:
            mc_count, short_count, code_count, long_count = 1, 0, 0, 0
        elif num_questions == 2:
            mc_count, short_count, code_count, long_count = 1, 1, 0, 0
        elif num_questions == 3:
            mc_count, short_count, code_count, long_count = 1, 1, 1, 0
        else:
            mc_count = max(1, int(num_questions * 0.4))
            short_count = max(1, int(num_questions * 0.3))
            code_count = max(0, int(num_questions * 0.2))
            long_count = max(0, num_questions - mc_count - short_count - code_count)
        
        prompt = f"""
As an expert AI Tutor, create a quiz about "{topic}" at {difficulty} difficulty level.
{search_context}
Generate exactly {num_questions} questions with the following distribution:
- {mc_count} multiple choice questions
- {short_count} short text response questions
- {code_count} code snippet questions (if applicable to the topic)
- {long_count} long essay questions

Format each question in JSON with this structure:
{{
    "question_id": number (starting from 1),
    "type": "multiple_choice" | "text_short" | "text_long" | "code",
    "question": "Full question text",
    "options": ["A", "B", "C", "D"] (only for multiple_choice),
    "correct_answer": "the correct answer",
    "explanation": "detailed explanation of the correct answer",
    "points": 10
}}

For multiple choice questions:
- Make options clear and distinct
- Include one clearly correct answer
- Add plausible distractors

For text questions:
- Ask for specific concepts or applications
- Include key terms in the correct answer
- Make evaluation criteria clear

For code questions (only if relevant to {topic}):
- Focus on practical implementation
- Include syntax and logic requirements
- Consider edge cases

Return ONLY a valid JSON array of questions, no additional text."""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                quiz_data = json.loads(json_match.group())
            else:
                # Fallback: try to parse the entire response as JSON
                quiz_data = json.loads(response_text)
            
            # Validate and fix question IDs
            for i, question in enumerate(quiz_data):
                question["question_id"] = i + 1
                if "points" not in question:
                    question["points"] = 10
            
            return {
                "questions": quiz_data,
                "topic": topic,
                "difficulty": difficulty,
                "created_at": datetime.now().isoformat()
            }
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse quiz JSON: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Quiz generation error: {str(e)}")
    
    def evaluate_answer(self, question: Dict, user_answer: str) -> Dict:
        """Evaluate a user's answer using AI"""
        
        prompt = f"""
Evaluate the user's answer for the following question:

Question: {question['question']}
Type: {question['type']}
Correct Answer: {question['correct_answer']}
User Answer: {user_answer}

Return evaluation in JSON format:
{{
    "is_correct": true/false,
    "score": 0-{question['points']},
    "feedback": "detailed feedback",
    "similarity_score": 0-100 (for text answers, null for multiple choice)
}}

Evaluation criteria:
- Multiple choice: Exact match required (case-insensitive)
- Text answers: Assess meaning, completeness, and key concepts (partial credit allowed)
- Code: Evaluate logic, syntax, and functionality (partial credit for working parts)
- Provide constructive feedback for improvement

Return ONLY the JSON object, no additional text."""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                # Fallback: try to parse the entire response as JSON
                evaluation = json.loads(response_text)
            
            # Ensure required fields exist
            if "is_correct" not in evaluation:
                evaluation["is_correct"] = evaluation.get("score", 0) > 0
            if "score" not in evaluation:
                evaluation["score"] = question["points"] if evaluation.get("is_correct", False) else 0
            if "feedback" not in evaluation:
                evaluation["feedback"] = "Answer evaluated."
                
            return evaluation
        except json.JSONDecodeError as e:
            # Fallback evaluation
            return {
                "is_correct": False,
                "score": 0,
                "feedback": f"Could not evaluate answer properly: {str(e)}",
                "similarity_score": None
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

# Dependency to validate API key
def get_tutor(api_key: str = Header(..., description="Gemini API Key")):
    try:
        return AITutor(api_key)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Tutor Quiz Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate_quiz": "POST /quiz/generate",
            "evaluate_answer": "POST /quiz/evaluate-answer",
            "evaluate_quiz": "POST /quiz/evaluate",
            "get_quiz": "GET /quiz/{quiz_id}",
            "list_quizzes": "GET /quiz",
            "delete_quiz": "DELETE /quiz/{quiz_id}"
        },
        "note": "Search functionality is optional and may timeout. Set use_search=true in quiz generation if needed."
    }

@app.post("/quiz/generate")
async def generate_quiz(
    request: QuizRequest,
    tutor: AITutor = Depends(get_tutor)
):
    """Generate a new quiz"""
    try:
        quiz_data = tutor.generate_quiz(
            request.topic,
            request.difficulty,
            request.num_questions,
            request.use_search
        )
        quiz_id = f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        quizzes[quiz_id] = quiz_data
        return {"quiz_id": quiz_id, **quiz_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quiz/{quiz_id}")
async def get_quiz(quiz_id: str):
    """Get quiz details"""
    if quiz_id not in quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return {"quiz_id": quiz_id, **quizzes[quiz_id]}

@app.post("/quiz/evaluate-answer")
async def evaluate_answer(
    request: AnswerRequest,
    tutor: AITutor = Depends(get_tutor)
):
    """Evaluate a single answer"""
    if request.quiz_id not in quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = quizzes[request.quiz_id]
    question = next((q for q in quiz["questions"] if q["question_id"] == request.question_id), None)
    
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    evaluation = tutor.evaluate_answer(question, request.user_answer)
    return evaluation

@app.post("/quiz/evaluate")
async def evaluate_quiz(
    request: QuizEvaluationRequest,
    tutor: AITutor = Depends(get_tutor)
):
    """Evaluate all answers in a quiz"""
    if request.quiz_id not in quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = quizzes[request.quiz_id]
    evaluations = {}
    total_score = 0
    max_score = sum(q["points"] for q in quiz["questions"])
    correct_count = 0
    
    for question in quiz["questions"]:
        q_id = question["question_id"]
        if q_id in request.answers:
            evaluation = tutor.evaluate_answer(question, request.answers[q_id])
            evaluations[q_id] = evaluation
            total_score += evaluation["score"]
            if evaluation["is_correct"]:
                correct_count += 1
        else:
            # No answer provided
            evaluations[q_id] = {
                "is_correct": False,
                "score": 0,
                "feedback": "No answer provided",
                "similarity_score": None
            }
    
    percentage = (total_score / max_score * 100) if max_score > 0 else 0
    grade = 'A' if percentage >= 90 else 'B' if percentage >= 80 else 'C' if percentage >= 70 else 'D' if percentage >= 60 else 'F'
    
    return QuizResultsResponse(
        quiz_id=request.quiz_id,
        total_score=total_score,
        max_score=max_score,
        percentage=round(percentage, 2),
        grade=grade,
        correct_count=correct_count,
        total_questions=len(quiz["questions"]),
        evaluations=evaluations
    )

@app.get("/quiz")
async def list_quizzes():
    """List all quizzes"""
    return {
        "quizzes": [
            {
                "quiz_id": quiz_id,
                "topic": quiz["topic"],
                "difficulty": quiz["difficulty"],
                "created_at": quiz["created_at"],
                "num_questions": len(quiz["questions"])
            }
            for quiz_id, quiz in quizzes.items()
        ],
        "total_quizzes": len(quizzes)
    }

@app.delete("/quiz/{quiz_id}")
async def delete_quiz(quiz_id: str):
    """Delete a quiz"""
    if quiz_id not in quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    del quizzes[quiz_id]
    return {"message": f"Quiz {quiz_id} deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_quizzes": len(quizzes)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)