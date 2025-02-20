@echo off

cd /d %~dp0

git --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Git is not installed.
    echo [INFO] Installing Git via winget...
    winget install --id Git.Git -e --source winget
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Git installation via winget failed.
        echo [INFO] Please install Git manually or ensure winget is available.
        pause
        exit /b
    )
    echo [INFO] Git installed successfully.
    echo.
    echo [INFO] You may need to close and reopen your terminal so that Git is available.
    echo [INFO] After reopening, re-run this script.
    pause
) ELSE (
    echo [INFO] Git is already installed.
)

for /f "tokens=*" %%i in ('pyenv --version 2^>null') do set PYENV_VAR==%%i
IF defined PYENV_VAR (
    echo [INFO] pyenv-win is not installed.
    echo [INFO] Installing pyenv-win...

    git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv\pyenv-win
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to clone the pyenv-win repository.
        pause
    )

    echo [INFO] pyenv-win has been installed.
    set "PATH=%USERPROFILE%\.pyenv\pyenv-win\bin;%USERPROFILE%\.pyenv\pyenv-win\shims;%PATH%"
) ELSE (
    echo [INFO] pyenv-win is already installed.
    set "PATH=%USERPROFILE%\.pyenv\pyenv-win\bin;%USERPROFILE%\.pyenv\pyenv-win\shims;%PATH%"
)

pyenv versions | findstr "3.11.9"
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Python 3.11.9 is not installed. Installing...
    pyenv install 3.11.9
    pyenv global 3.11.9
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install Python 3.11.9.
        pause
    )
) ELSE (
    echo [INFO] Python 3.11.9 is already installed.
)

poetry --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Poetry is not installed.
    echo [INFO] Installing Poetry now...

    curl -sSL https://install.python-poetry.org -o install-poetry.py
    IF %ERRORLEVEL% NEQ 0 (
         echo [ERROR] Failed to download the Poetry installer.
         pause
    )

    python install-poetry.py
    IF %ERRORLEVEL% NEQ 0 (
         echo [ERROR] Failed to install Poetry.
         pause
    )

    del install-poetry.py
    set "PATH=%USERPROFILE%\AppData\Roaming\Python\Scripts;%PATH%"
    echo [INFO] Poetry has been installed.
) ELSE (
    echo [INFO] Poetry is already installed.
)

poetry run python predict.py
