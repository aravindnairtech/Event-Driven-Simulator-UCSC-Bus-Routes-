from collections import deque # Used to implement queues.
import random # Random choice, etc.
import heapq # Used in discrete event simulator
import numpy as np # Used for gamma probability distribution, and percentiles.
import matplotlib.pyplot as plt
import itertools
from tabulate import tabulate # To display the bus status.


def fmt(x):
    """Formats a number x which can be None, for convenience."""
    return None if x is None else "{:.2f}".format(x)

class Event(object):

    def __init__(self, method, delay=0, args=None, kwargs=None):
        """An event consists in calling a specified method after a delay,
        with a given list of args and kwargs."""
        self.method = method
        self.delay = delay
        self.args = args or []
        self.kwargs = kwargs or {}
        self.time = None # Not known until this is added to the queue.

    def __call__(self, time):
        """This causes the event to happen, returning a list of new events
        as a result. Informs the object of the time of occurrence."""
        return self.method(*self.args, time=time, **self.kwargs)

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return "@{}: {} {} {} {} dt={:.2f}".format(
            fmt(self.time),
            self.method.__self__.__class__.__name__,
            self.method.__name__,
            self.args, self.kwargs, self.delay
        )

      
class EventSimulator(object):

    def __init__(self, trace=False):
        self.events = []
        self.time = 0 # This is the global time.
        self.trace = trace

    def add_event(self, event):
        """Adds an event to the queue."""
        event.time = self.time + event.delay
        heapq.heappush(self.events, event)

    def step(self):
        """Performs a step of the simulation."""
        if len(self.events) > 0:
            event = heapq.heappop(self.events)
            self.time = event.time
            new_events = event(self.time) or []
            for e in new_events:
                self.add_event(e)
            if self.trace:
                print("Processing:", event)
                print("New events:", new_events)
                print("Future events:", self.events)

    def steps(self, number=None):
        """Performs at most number steps (or infinity, if number is None)
        in the simulation."""
        num_done = 0
        while len(self.events) > 0:
            self.step()
            num_done += 1
            if num_done == number:
                break
class Person(object):

    def __init__(self, start_time, source, destination, have_arrived,
                 person_id=None):
        """
        @param start_time: time at which a person enters the system.
        @param source: stop at which the person wants to climb on.
        @param destination: destination stop.
        @param have_arrived: list of people who have arrived, so we can
            plot their bus time.
        """
        self.start_time = start_time
        self.bus_time = None # Time at which climbed on bus
        self.end_time = None
        self.source = source
        self.destination = destination
        self.have_arrived = have_arrived
        # for id purpose
        self.id = person_id

    # Event method
    def arrived(self, time=None):
        """The person has arrived to their destination."""
        self.end_time = time
        self.have_arrived.append(self)
        return [] # No events generated as a consequence.

    def start_bus(self, time=None):
        """The person starts getting on the bus."""
        self.bus_time = time

    @property
    def elapsed_time(self):
        return None if self.end_time is None else self.end_time - self.start_time

    @property
    def travel_time(self):
        return None if self.end_time is None else self.end_time - self.bus_time

    @property
    def wait_time(self):
        return None if self.end_time is None else self.bus_time - self.start_time

    def __repr__(self):
        return f"Person #: {self.id}, source: {self.source}, dest: {self.destination}"
class Source(object):
    """Creates people, and adds them to the queues."""

    def __init__(self, rate=1., queue_ring=None, number=None, have_arrived=None):
        """
        @param rate is the rate at which people are generated.
        @param number is the total number of people to generate; None = unlimited.
        @param queue_ring is the queue ring (a list of queues) where people are added.
        @param have_arrived is the list where people who have arrived are added.
        """
        self.rate = rate
        self.queue_ring = queue_ring
        self.num_stops = len(queue_ring)
        self.number = number
        self.have_arrived = have_arrived
        self.person_id = 0 # For debugging.

    # Event method
    def start(self, time=None):
        if self.number == 0:
            return [] # Nothing more to be done.
        # Creates the person
        self.person_id += 1
        source, destination = random.sample(range(self.num_stops), 2)
        person = Person(time, source, destination, self.have_arrived,
                        person_id = self.person_id)
        queue = self.queue_ring[source]
        enter_event = Event(queue.enter, args=[person])
        # Schedules the next person creation.
        self.number = None if self.number is None else self.number - 1
        dt = np.random.gamma(1, 1/self.rate)
        start_event = Event(self.start, delay=dt)
        return [enter_event, start_event]
### Class Queue

class Queue(object):

    def __init__(self):
        """We create a queue."""
        
        self.people = []

    # Event method
    def enter(self, person, time=None):
       
        self.people.append(person)
        #return [Event(self.enter)]

    ### You can put here any other methods that might help you.
    
    def nextinqueue(self):
        try:
            return self.people.pop(0)
        except IndexError:
            return None

    #def people(self):
    #    return list(self.waitq)



#@ title Class Bus

class Bus(object):

    def __init__(self, queue_ring, max_capacity, geton_time, nextstop_time,
                 bus_id=None):
        """The bus is created with the following parameters:
        @param max_capacity: the max capacity of the bus.
        @param queue_ring: the ring (list) of queues representing the stops.
        @param geton_time: the expected time that it takes for a person to climb
            the 2 steps to get on the bus.  The time a person takes to get on is
            given by np.random.gamma(2, geton_time / 2).
            This is the same as the time to get off the bus.
        @param nextstop_time: the average time the bus takes to go from one stop
            to the next.  The actual time is given by
            np.random.gamma(10, nextstop_time/10).
        @param bus_id: An id for the bus, for debugging.
        """
        self.queue_ring = queue_ring
        self.max_capacity = max_capacity
        self.geton_time = geton_time
        self.nextstop_time = nextstop_time
        self.id = bus_id
        ### Put any other thing you need in the initialization below.
        
        self.current_stop = None
        self.passengers = []


    @property
    def stop(self):
        """Returns the current (most recent) stop of the bus,
        as an integer."""
        
        return self.current_stop

    @property
    def onboard(self):
        """Returns the list of people on the bus."""
      
        return self.passengers

    @property
    def occupancy(self):
        """Returns the number of passengers on the bus."""
    
        return len(list(self.passengers))

    # Event method.
    def arrive(self, stop_idx, time=None):
        """Arrives at the next stop."""
        ### You can do what you want here.
       
        traveltime = np.random.gamma(10, self.nextstop_time/10)
        boardingtime = 0
        self.current_stop = stop_idx
        passangerq = self.queue_ring[self.current_stop]
        passengeronboard = self.occupancy
        exitingpassenger = self.peekpassengerlist()
        while exitingpassenger != None:
            passengeronboard -= 1
            boardingtime += np.random.gamma(2, self.geton_time / 2)
            exitingpassenger.arrived(time)
            exitingpassenger = self.peekpassengerlist()
        while passengeronboard < self.max_capacity:
            passengertoenter = passangerq.nextinqueue()
            if(passengertoenter!= None):
                boardingtime += np.random.gamma(2, self.geton_time / 2)
                passengertoenter.start_bus(time)
                self.passengers.append(passengertoenter)
                passengeronboard +=1
            else:
                break
        self.passengers.sort(key = lambda x: x.destination, reverse = False)

        if self.current_stop < len(self.queue_ring) - 1:
            next_stop = self.current_stop +1
        else:
            next_stop = 0
        print(f'==>{boardingtime} => {traveltime}')
        return [Event(self.arrive, args=[next_stop],delay=traveltime+boardingtime)]

    def __repr__(self):
        """This will print a bus, which helps in debugging."""
        return "Bus#: {}, #people: {}, dest: {}".format(
            self.id, self.occupancy, [p.destination for p in self.onboard])



    ### You can have as many other methods as you like, including other
    ### events for the bus.  Up to you.

    def peekpassengerlist(self):
        if(len(self.passengers) == 0):
            return None
        for passenger in self.passengers:
            if passenger.destination == self.current_stop:
                self.passengers.remove(passenger)
                return passenger
        return None
        # passenger = self.passengers[0]
        # if passenger.destination == self.current_stop:
        #     del self.passengers[0]
        #     return passenger
        # return None

    # def __lt__(self, other):
    #     return self.destination < other.destination


def bus_distance(ix, iy, num_stops=20):
    """Returns the distance between two buses."""
    if ix is None or iy is None:
        return None
    d1 = (ix - iy + num_stops) % num_stops
    d2 = (iy - ix + num_stops) % num_stops
    return min(d1, d2)

 class Simulation(object):

    def __init__(self, num_stops=20, num_buses=1,
                 bus_nextstop_time=1, bus_geton_time=0.1,
                 bus_max_capacity=50,
                 person_rate=2, destinations="random",
                 number_of_people=None,
                 trace=False):
        self.num_stops = num_stops
        self.num_buses = num_buses
        self.bus_max_capacity = bus_max_capacity
        # Chooses the initial stops for the buses.
        self.initial_stops = list(np.mod(np.arange(0, self.num_buses) * max(1, num_stops // num_buses), num_stops))
        # Speeds
        self.bus_nextstop_time = bus_nextstop_time
        self.bus_geton_time = bus_geton_time
        self.person_rate = person_rate
        # Event simulator
        self.simulator = EventSimulator(trace=trace)
        # Builds the queue ring
        self.queue_ring = [Queue() for _ in range(num_stops)]
        # And the source.
        self.have_arrived = []
        self.source = Source(rate=person_rate, queue_ring=self.queue_ring,
                             number=number_of_people, have_arrived=self.have_arrived)
        # And the buses.
        self.buses = [Bus(queue_ring=self.queue_ring,
                          max_capacity=bus_max_capacity,
                          geton_time=bus_geton_time,
                          nextstop_time=bus_nextstop_time,
                          bus_id=i + 1)
            for i in range(num_buses)]
        # We keep track of the distances between buses, and the
        # bus occupancies.
        self.positions = [[] for _ in range(num_buses)]
        self.occupancies = [[] for _ in range(num_buses)]


    def start(self):
        """Starts the simulation."""
        # Injects the initial events in the simulator.
        # Source.
        self.simulator.add_event(Event(self.source.start))
        # Buses.
        for i, bus in enumerate(self.buses):
            self.simulator.add_event(
                Event(bus.arrive, args=[self.initial_stops[i]]))

    def step(self):
        """Performs a step in the simulation."""
        self.simulator.step()
        for bus_idx in range(self.num_buses):
            self.positions[bus_idx].append(self.buses[bus_idx].stop)
            self.occupancies[bus_idx].append(self.buses[bus_idx].occupancy)

    def plot(self):
        """Plots the history of positions and occupancies."""
        # Plots positions.
        for bus_idx in range(self.num_buses):
            plt.plot(self.positions[bus_idx])
        plt.title("Positions")
        plt.show()
        # Plots occupancies.
        for bus_idx in range(self.num_buses):
            plt.plot(self.occupancies[bus_idx])
        plt.title("Occupancies")
        plt.show()
        # Plots times.
        plt.hist([p.wait_time for p in self.have_arrived])
        plt.title("Wait time")
        plt.show()
        plt.hist([p.travel_time for p in self.have_arrived])
        plt.title("Time on the bus")
        plt.show()
        plt.hist([p.elapsed_time for p in self.have_arrived])
        plt.title("Total time")
        plt.show()
        # Plots bus distances
        if self.num_buses > 1:
            for i, j in itertools.combinations(range(self.num_buses), 2):
                ds = [bus_distance(pi, pj, num_stops=self.num_stops)
                      for pi, pj in zip(self.positions[i], self.positions[j])]
                plt.plot(ds)
            plt.title("Bus distances")
            plt.show()

    def status(self):
        """Tabulates the bus location and queue status."""
        headers = ["Stop Index", "Queue", "Buses"]
        rows = []
        for stop_idx, queue in enumerate(simulation.queue_ring):
            buses = [b for b in self.buses if b.stop == stop_idx]
            busStr = "\n".join([bus.__str__() for bus in buses])
            personStr = "\n".join([person.__str__() for person in queue.people])
            row = [f"{stop_idx}", f"{personStr}", f"{busStr}"]
            rows.append(row)
        print(tabulate(rows, headers, tablefmt="grid", stralign='left', numalign='right'))

simulation = Simulation(num_stops=5, num_buses=2, person_rate=2, trace=False)
simulation.start()
for i in range(30):
    simulation.step()
    print(f"\nState after step {i}")
    simulation.status()


                
