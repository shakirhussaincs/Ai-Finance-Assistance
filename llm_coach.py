"""
pipelines/llm_coach.py
Pipeline 4 — LLM Coaching (Gemini API via google-generativeai SDK)
Injects real ML outputs as structured context into Gemini.
Falls back to rule-based templates if API is unavailable.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_GEMINI_OK = False
try:
    import google.generativeai as genai
    _GEMINI_OK = True
except ImportError:
    logger.warning("google-generativeai package not installed — rule-based fallback active.")

SYSTEM_PROMPT = """You are a warm, expert AI personal finance coach. You have access to the user's real spending data analysed by ML models.

Your role:
- Translate structured ML insights into clear, personalized, actionable financial advice
- Speak like a knowledgeable friend, not a bank manager — encouraging but honest
- Always reference specific numbers from the context provided
- Keep responses focused: 3-5 key points maximum
- Pair every problem with a concrete next step
- Never invent data not provided in the context

User profile context will be provided in <financial_context> tags.
Current date: {current_date}
"""

RULE_BASED_TEMPLATE = """📊 **Your Financial Summary**

{spending_lines}

{anomaly_line}

{forecast_line}

{savings_line}

💡 **Quick Tips**
• Your top spending category is **{top_category}** — review if this aligns with your priorities.
• Set a monthly budget alert for categories that regularly overshoot.
• Consider automating savings transfers at the start of each month.
"""


def _rule_based(context: Dict[str, Any]) -> str:
    spending = context.get("spending_summary", {})
    spending_lines = "\n".join(
        f"• {cat}: ${amt:.0f}" for cat, amt in list(spending.items())[:6]
    ) if spending else "• No spending data available"

    anomalies    = context.get("anomalies", [])
    anomaly_line = (
        f"⚠️  **{len(anomalies)} anomalies detected** this period — check your Anomalies tab."
        if anomalies else "✅ No unusual transactions detected."
    )

    fc = context.get("next_month_forecast")
    forecast_line = (
        f"📈 **Forecast**: You're projected to spend **${fc:.0f}** next month."
        if fc else ""
    )

    gap = context.get("savings_gap", 0)
    savings_line = (
        f"🎯 You're **${gap:.0f} behind** your savings goal this month."
        if gap and gap > 0 else "✅ You're on track with your savings goal."
    )

    top_category = list(spending.keys())[0] if spending else "N/A"

    return RULE_BASED_TEMPLATE.format(
        spending_lines=spending_lines,
        anomaly_line=anomaly_line,
        forecast_line=forecast_line,
        savings_line=savings_line,
        top_category=top_category,
    )


def _build_context_block(context: Dict[str, Any]) -> str:
    """Format ML outputs into a structured XML block for Gemini."""
    sections = []

    # Spending summary
    spending = context.get("spending_summary", {})
    if spending:
        lines = ["<spending_summary>"]
        for cat, amt in spending.items():
            lines.append(f"  {cat}: ${amt:.2f}")
        lines.append("</spending_summary>")
        sections.append("\n".join(lines))

    # Anomalies
    anomalies = context.get("anomalies", [])
    if anomalies:
        lines = ["<anomalies>"]
        for a in anomalies[:5]:
            lines.append(
                f"  [{a.get('severity','?')}] ${a.get('amount',0):.2f} "
                f"at {a.get('category','?')} | z={a.get('z_score',0):.1f}σ | {a.get('date','')}"
            )
        lines.append("</anomalies>")
        sections.append("\n".join(lines))

    # Forecasts
    forecasts = context.get("forecasts", [])
    if forecasts:
        lines = ["<spending_forecasts>"]
        for fc in forecasts[:8]:
            lines.append(
                f"  {fc.get('category','?')} | {fc.get('month','?')}: "
                f"${fc.get('yhat',0):.0f} [${fc.get('yhat_lower',0):.0f}–${fc.get('yhat_upper',0):.0f}]"
            )
        lines.append("</spending_forecasts>")
        sections.append("\n".join(lines))

    # Monthly history
    history = context.get("monthly_history", [])
    if history:
        lines = ["<monthly_history_last_6>"]
        for h in history[-6:]:
            lines.append(f"  {h.get('month','?')}: ${h.get('total',0):.0f}")
        lines.append("</monthly_history_last_6>")
        sections.append("\n".join(lines))

    # User profile
    profile = context.get("user_profile", {})
    if profile:
        lines = ["<user_profile>"]
        if profile.get("monthly_income"):
            lines.append(f"  Monthly income: ${profile['monthly_income']}")
        if profile.get("savings_goal"):
            lines.append(f"  Monthly savings goal: ${profile['savings_goal']}")
        if profile.get("savings_gap"):
            lines.append(f"  Current savings gap: ${profile['savings_gap']:.0f}")
        lines.append("</user_profile>")
        sections.append("\n".join(lines))

    return "<financial_context>\n" + "\n\n".join(sections) + "\n</financial_context>"


class LLMCoach:
    """
    Gemini-powered financial coaching with multi-turn memory.
    Falls back to rule-based templates if API unavailable.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 max_tokens: int = 1024):
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = None

        if _GEMINI_OK and api_key and api_key != "your_key_here":
            try:
                genai.configure(api_key=api_key)
                self._client = True # Just a flag to show we are configured
                logger.info(f"[LLMCoach] Gemini client ready ({model})")
            except Exception as e:
                logger.error(f"[LLMCoach] Failed to init client: {e}")
        else:
            logger.info("[LLMCoach] Running in rule-based fallback mode")

    @property
    def using_gemini(self) -> bool:
        return self._client is not None

    def chat(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a coaching response.

        Args:
            user_message  : The user's question/request.
            context       : ML insights dict (spending, anomalies, forecasts).
            history       : [{role, content}, ...] prior conversation turns.
            user_profile  : {monthly_income, savings_goal}.

        Returns:
            {response: str, tokens_used: int, model: str}
        """
        context = context or {}
        if user_profile:
            context["user_profile"] = user_profile

        if self._client is None:
            return {
                "response":    _rule_based(context),
                "tokens_used": 0,
                "model":       "rule_based",
            }

        system = SYSTEM_PROMPT.format(current_date=datetime.now().strftime("%B %d, %Y"))

        # Build messages list
        messages = []

        # Replay conversation history (last 10 turns)
        for turn in (history or [])[-10:]:
            role = "model" if turn["role"] in ["assistant", "model"] else "user"
            messages.append({"role": role, "parts": [turn["content"]]})

        # Current user message with injected ML context
        if context:
            context_block = _build_context_block(context)
            full_msg = f"{context_block}\n\nUser question: {user_message}"
        else:
            full_msg = user_message

        try:
            model_instance = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system
            )
            chat_session = model_instance.start_chat(history=messages)
            
            response = chat_session.send_message(
                full_msg,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                )
            )
            
            try:
                tokens_used = model_instance.count_tokens(chat_session.history).total_tokens
            except:
                tokens_used = 0

            return {
                "response":    response.text,
                "tokens_used": tokens_used,
                "model":       self.model,
            }
        except Exception as e:
            logger.error(f"[LLMCoach] API error: {e}")
            return {
                "response":    _rule_based(context),
                "tokens_used": 0,
                "model":       "rule_based_fallback",
            }

    def monthly_report(
        self,
        context: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate an unsolicited monthly financial health report."""
        prompt = (
            "Generate a concise monthly financial health report. "
            "Structure it as: (1) Health score /10 with 1-line rationale, "
            "(2) Top 2 wins this month, (3) Top 2 areas to improve, "
            "(4) ONE specific action to take before next month."
        )
        return self.chat(prompt, context=context, user_profile=user_profile)
