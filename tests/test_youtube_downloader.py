import pytest
from unittest.mock import patch, Mock
from quint.data.youtube import download_youtube_video


def mock_youtube_stream():
    mock_stream = Mock()
    mock_stream.download.return_value = "/path/to/downloaded/file.mp4"
    return [mock_stream]


def test_download_youtube_video():
    with patch("quint.data.youtube.YouTube", autospec=True) as mock_youtube:
        mock_instance = mock_youtube.return_value
        mock_instance.streams.filter.return_value = mock_youtube_stream()

        result = download_youtube_video("test_video_id")
        assert result == "file.mp4"  # based on the mocked return value

        # Test the case where no audio streams are found
        mock_instance.streams.filter.return_value = []
        with pytest.raises(Exception, match="No audio streams found for video:"):
            download_youtube_video("test_video_id")

        # Test fetching error
        mock_youtube.side_effect = Exception("Fetching error")
        with pytest.raises(Exception, match="Error fetching video from URL:"):
            download_youtube_video("test_video_id")


# other test cases can be added as needed
