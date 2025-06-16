-- Таблица кандидатов
CREATE TABLE candidate (
    id               SERIAL PRIMARY KEY,
    created_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    rating           REAL    NOT NULL,
    education        VARCHAR(50)  NOT NULL,
    specialty        VARCHAR(100) NOT NULL,
    city             VARCHAR(100) NOT NULL,
    gender           VARCHAR(10),
    rubbish_col      VARCHAR(100),
    skill_set_raw    TEXT,
    experience_years INT,
    resume_url       VARCHAR(255),
    INDEX idx_rating (rating)
);

-- Справочник навыков
CREATE TABLE skill (
    id        SERIAL PRIMARY KEY,
    name      VARCHAR(100) UNIQUE NOT NULL
);

-- Связующая таблица многие-ко-многим кандидатов и навыков
CREATE TABLE candidate_skill (
    candidate_id INT NOT NULL REFERENCES candidate(id),
    skill_id     INT NOT NULL REFERENCES skill(id),
    PRIMARY KEY (candidate_id, skill_id)
);

-- Эксперименты
CREATE TABLE experiment (
    id             SERIAL PRIMARY KEY,
    run_at         TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata       JSON,
    hyperparams    JSON
);

-- Выборки эксперимента
CREATE TABLE experiment_sample (
    id             SERIAL PRIMARY KEY,
    experiment_id  INT NOT NULL REFERENCES experiment(id),
    candidate_id   INT NOT NULL REFERENCES candidate(id),
    label          BOOLEAN NOT NULL,
    features       JSONB    NOT NULL
);

-- Метрики эксперимента
CREATE TABLE experiment_metric (
    id             SERIAL PRIMARY KEY,
    experiment_id  INT NOT NULL REFERENCES experiment(id),
    name           VARCHAR(50) NOT NULL,
    value          REAL       NOT NULL,
    recorded_at    TIMESTAMP  NOT NULL DEFAULT NOW(),
    INDEX idx_name (name)
);