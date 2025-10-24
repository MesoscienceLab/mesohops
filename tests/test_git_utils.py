import pytest
from unittest.mock import patch, MagicMock
from mesohops.util.git_utils import get_git_commit_hash


def test_get_git_commit_hash_success():
    """Test that get_git_commit_hash returns a valid hash in a git repository."""
    # Since we're running in a git repository, we should get a valid hash
    result = get_git_commit_hash()

    # Check that the result is a non-empty string (a valid git hash)
    assert isinstance(result, str)
    assert len(result) > 0

    # Verify it's a valid git hash format (40 hex characters)
    assert all(c in '0123456789abcdef' for c in result.lower())
    assert len(result) == 40


@patch('subprocess.run')
def test_get_git_commit_hash_command_error(mock_run):
    """Test that get_git_commit_hash returns error message when git command fails."""
    # Mock the subprocess.run to return a generic error
    mock_process = MagicMock()
    mock_process.returncode = 1  # Generic error code
    mock_process.stderr = "A gigantic walrus ate the git repository"
    mock_run.return_value = mock_process

    # Call the function and check the result
    result = get_git_commit_hash()
    assert result == "A gigantic walrus ate the git repository"


@patch('subprocess.run')
def test_get_git_commit_hash_file_not_found(mock_run):
    """Test that get_git_commit_hash handles FileNotFoundError correctly."""
    # Mock subprocess.run to raise FileNotFoundError
    mock_run.side_effect = FileNotFoundError

    # Call the function and check the result
    result = get_git_commit_hash()

    # The function should return string when FileNotFoundError is caught
    assert result == "Git is not installed or not found in PATH."
