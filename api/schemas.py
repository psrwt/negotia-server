from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class Product(BaseModel):
    id: str = Field(..., alias="ID")
    company_name: str = Field(..., alias="Company Name")
    model_name: str = Field(..., alias="Model Name")
    max_price: int = Field(..., alias="Max Price")
    capacity: int = Field(..., alias="Capacity")
    ram: int = Field(..., alias="ram")
    back_camera: str = Field(..., alias="Back Camera")
    front_camera: str = Field(..., alias="Front Camera")
    processor: str = Field(..., alias="Processor")
    screen_size: str = Field(..., alias="Screen Size")
    battery: int = Field(..., alias="battery")
    description: str = Field(..., alias="Text")
    image_url: Optional[str] = None

class SpecialDeal(BaseModel):
    heading: str
    deal_price: float
    products_involved: List[Product]

class ChatRequest(BaseModel):
    user_message: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    language: str = Field(default="en-US")  # Language code
    voice_id: Optional[str] = None  # Optional voice ID override

class ChatResponse(BaseModel):
    text: str
    audio_url: Optional[str] = None
    products: List[Product] = Field(default_factory=list)
    special_deal: Optional[SpecialDeal] = None

class FinalAnswer(BaseModel):
    text: str = Field(description="The conversational text to display to the user.")
    product_ids: List[str] = Field(default_factory=list)
    deal_heading: Optional[str] = None
    deal_price: Optional[float] = None
    deal_product_ids: Optional[List[str]] = Field(default_factory=list)

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None   # optional override; else use env MURF_VOICE_ID
    language: Optional[str] = None   # language code for voice selection
    format: Optional[str] = "MP3"    # MP3, WAV, FLAC, ALAW, ULAW, PCM, OGG
    sample_rate: Optional[int] = 44100
    style: Optional[str] = None      # e.g., "Conversational"
    encode_as_base64: Optional[bool] = False  # set True if you want audio in response