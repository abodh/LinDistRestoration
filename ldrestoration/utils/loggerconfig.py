import logging

def setup_logging() -> None:
    """Configure the logging for the entire package
    """
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        # filename='outputlog.log')
    
setup_logging()