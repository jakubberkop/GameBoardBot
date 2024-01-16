            # if not done:
            #     target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            # output = self.forward(state_tensor)
            # target_f = output.clone()
            # 
            # print("WTF does the argmax doing here?", np.argmax(action))
            # target_f[0][np.argmax(action)] = target
            # target_f.detach()