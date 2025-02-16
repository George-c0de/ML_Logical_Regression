from pydantic_settings import BaseSettings


class Config(BaseSettings):
    DBNAME: str
    HOST: str
    USER: str
    PASSWORD: str
    PORT: int
