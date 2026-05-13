export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  agent?: string;
  image?: string;
  resultImage?: string;
  timestamp: Date;
  isHumanValidationRequired?: boolean;
}

export interface ApiResponse {
  status: string;
  response: string;
  agent: string;
  result_image?: string;
  message?: string; // For validation responses
}
