import argparse
from src import train_reinforce, train_ppo, train_trpo, train_a2c

def main():
    parser = argparse.ArgumentParser(description='DRL Policy Gradient Optimization Experiment')
    parser.add_argument('--algorithm', type=str, default='reinforce',
                        help='Algorithm to run: reinforce, ppo, trpo, a2c')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use: sgd, adam, rmsprop')
    args = parser.parse_args()
    
    alg = args.algorithm.lower()
    if alg == 'reinforce':
        train_reinforce.train(args.optimizer)
    elif alg == 'ppo':
        train_ppo.train(args.optimizer)
    elif alg == 'trpo':
        train_trpo.train(args.optimizer)
    elif alg == 'a2c':
        train_a2c.train(args.optimizer)
    else:
        print("Invalid algorithm specified. Choose from: reinforce, ppo, trpo, a2c")
    
if __name__ == '__main__':
    main()