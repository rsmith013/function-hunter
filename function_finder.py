#!/usr/bin/env python
import pika
import uuid
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from configparser import ConfigParser


def fit_sin(xx, yy):
    """
    Fit sin to the input time sequence, and return fitting parameters
    "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc

    :param xx: x array
    :param yy: y array
    :return: Dictionary containing the descriptors of the best fit curve
    with keys:
    amp: Amplitude
    omega: Angular Frequency
    phase: Point (in radians) in the oscillation cycle where the wave is at x=0
    offset: Offset from the origin
    freq: Oscillations/sec
    period: Time to complete one cycle
    fitfunc: Function which describes the best fit curve
    maxcov: Maximum covariance of the best fit curve
    rewresult: The raw values returned from the fitting function
    """

    # Convert the input values to numpy arrays
    xx = np.array(xx)
    yy = np.array(yy)

    # Use FFT to get the wave frequency assume uniform spacing
    ff = np.fft.fftfreq(len(xx), (xx[1] - xx[0]))
    Fyy = abs(np.fft.fft(yy))

    # Extract the peak excluding the zero frequency "peak", which is related to offset
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])

    # Get an approximation for the amplitude using the standard deviation
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)

    # Generate the first guess
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w * t + p) + c

    # Fit the first guess to the curve
    popt, pcov = scipy.optimize.curve_fit(sinfunc, xx, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2. * np.pi)

    #  Create the function to apply and recreate the sine wave
    fitfunc = lambda t: A * np.sin(w * t + p) + c

    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
            "maxcov": np.max(pcov), "rawresult": (guess, popt, pcov)}


class Config(object):
    def __init__(self, filename):
        # Load config file
        config = ConfigParser()
        config.read(filename)

        self.host = config.get('SERVER', 'host')
        self.port = int(config.get('SERVER', 'port'))
        self.auth = (config.get('SERVER', 'username'),
                     config.get('SERVER', 'password'))
        self.send_q = config.get('QUEUES', 'send_queue')
        self.reply_q = config.get('QUEUES', 'reply_queue')
        self.max_retry = int(config.get('SERVER', 'max_retry'))


class RpcClient(object):
    """
    Send and receive messages via a RabbitMQ server
    """

    def __init__(self, host, reply_q, auth=None, port=5672, max_retry=10):
        """
        :param host: Host IP
        :param reply_q: Queue to process messages from
        :param auth: Tuple containing authentication params(username,password) default: None
        :param port: Host Port default: 5672
        :param max_retry: Max no. messages to read before retrying default: 10

        """
        if auth is not None:
            username, password = auth
            credentials = pika.PlainCredentials(username, password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, port=port, credentials=credentials))
        else:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, port=port))

        self.max_retry = max_retry

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue=reply_q)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)
        self.attempt_count = 0

        # Params init for the call and response
        self.response = None
        self.corr_id = ''

    def on_response(self, ch, method, props, body):
        """
        Callback to process messages read from the queue

        :param ch: Message channel properties
        :param method: Message method properties
        :param props: Message properties object
        :param body: Message body
        """

        if self.corr_id == props.correlation_id:
            self.response = body
        elif self.attempt_count < self.max_retry:
            self.attempt_count += 1
        else:
            self.response = {}

    def call(self, n, send_q):
        """
        Send a message to the queue. The message is a JSON object {'num':n}

        :param n: Real number to send to queue
        :param send_q: Queue to send the message to
        :return: Return matching response from server
        """
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key=send_q,
                                   properties=pika.BasicProperties(
                                       correlation_id=self.corr_id
                                   ),
                                   body=json.dumps({'num': n}))
        while self.response is None:
            self.connection.process_data_events()
        return self.response


def main():
    """
    Main script
    """
    # Get config
    conf = Config('config.ini')

    # Initialize the connection
    rpc = RpcClient(host=conf.host,
                    reply_q=conf.reply_q,
                    auth=conf.auth,
                    port=conf.port,
                    max_retry=conf.max_retry)

    # Prepare the array to store the call and response
    x = []
    y = []

    # Generate an array to send to the server
    for i in np.linspace(0, 10, 101):
        response = {}

        # Check for a non-empty dict. If the dict is empty, the client could not match
        # the message
        while not response:
            response = rpc.call(i, conf.send_q)

        # Parse the response and append to the relevant lists
        json_response = json.loads(response)
        x.append(i)
        y.append(json_response.get('num'))

    # Get best fit curve and params to solve for the hidden function
    res = fit_sin(x, y)
    print(
        f'Amplitude: {res["amp"]:.2f} Omega: {res["omega"]:.2f} '
        f'Phase: {res["phase"]:.2f} Offset: {res["offset"]:.2f}')

    # Plot results to verify
    plt.plot(x, y, 'kx', label="Original")
    plt.plot(x, res['fitfunc'](np.array(x)), 'r-', label="Fitted curve")
    plt.plot(x, 2 * np.sin(x + np.full(len(x), 1.05)), 'g.', label="Validation")
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
