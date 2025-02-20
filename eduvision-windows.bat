@echo off

cd /d %~dp0

pyenv --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo pyenv-win is not installed.
    echo Installing pyenv-win...

    git --version >nul 2>&1
    IF ERRORLEVEL 1 (
        echo Git is not installed.
        echo Installing Git...

        winget install --id Git.Git -e --source winget
        IF ERRORLEVEL 1 (
             echo Failed to download the Git installer.
             pause
             exit /b
        )
        echo Git has been installed.
        start "" cmd /k "cd /d %CD%"
        exit
    ) ELSE (
        echo Git is already installed.
    )

    git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv\pyenv-win
    IF ERRORLEVEL 1 (
         echo Failed to clone the pyenv-win repository.
         pause
         exit /b
    )

    set "PATH=%USERPROFILE%\.pyenv\pyenv-win\bin;%USERPROFILE%\.pyenv\pyenv-win\shims;%PATH%"
    echo pyenv-win has been installed.

    echo pyenv-win has been installed.
) ELSE (
    echo pyenv-win is already installed.
)

pyenv versions | findstr "3.11.9" >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python 3.11.9 is not installed. Installing...
    pyenv install 3.11.9
    IF ERRORLEVEL 1 (
         echo Failed to install Python 3.11.9.
         pause
         exit /b
    )
) ELSE (
    echo Python 3.11.9 is already installed.
)

pyenv global 3.11.9
echo Python 3.11.9 has been set as the global version.
poetry --version >nul 2>&1

IF ERRORLEVEL 1 (
    echo Poetry is not installed.
    echo Installing Poetry now...

    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    IF ERRORLEVEL 1 (
         echo Failed to download the Poetry installer.
         pause
         exit /b
    )

    echo Poetry has been installed.
) ELSE (
    echo Poetry is already installed.
)
poetry run python predict.py

pause