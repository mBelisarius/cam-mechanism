import matplotlib as mpl
import matplotlib.animation as anm
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import CubicSpline

from utils import arg_closest, linear, to_cartesian, to_degrees


class Cam:
    def __init__(self, n, dist, path, constrained=False):
        self._points = n
        self._dist = dist
        self._path = path
        self._constrained = constrained
        self._data_max = np.zeros(n)
        self._data_min = np.zeros(n)

    @property
    def points(self):
        return self._points

    @property
    def path(self):
        return self._path

    @property
    def max_size(self):
        return self._data_max

    @property
    def min_size(self):
        return self._data_min

    @staticmethod
    def _config_subplot(axis, title):
        axis.text(1.02, 0.5, title, rotation=90, va='center', ha='left', transform=axis.transAxes)
        axis.grid()
        axis.set_xticks(np.arange(0., 361., 60.))
        axis.set_xlim([0., 360.])

    @staticmethod
    def _build_subplot_graph(axis, theta, reference, target_data, cam_data, color_data,
                             animated=False, theta_view=None):
        if animated and theta_view is None:
            raise ValueError('Parameter `theta_view` is necessary')

        theta_deg = to_degrees(theta)

        # data_view = np.interp(theta_view, theta, target_data)
        data_min = np.min(target_data)
        data_max = np.max(target_data)
        data_all_min = np.min([data_min] + [np.min(d) for d in cam_data])
        data_all_max = np.max([data_max] + [np.max(d) for d in cam_data])

        graph_axis = axis.plot(
            to_degrees(theta), np.full_like(theta, reference),
            linestyle='-', color='black'
        )[0]

        graph_target = axis.plot(
            theta_deg, target_data,
            linestyle=':', color='black'
        )[0]

        graph_follower = [
            axis.plot(
                theta_deg, d,
                linestyle='-', color=c,
            )[0] for d, c in zip(cam_data, color_data)
        ]

        graph_vert = axis.plot(
            2 * [theta_deg[0]], [data_all_min, data_all_max],
            linestyle='-', color='black'
        )[0] if animated else None

        graph_follower_pnt = [
            axis.plot(
                theta_deg[0], d[0],
                linestyle='', marker='o', color=c
            )[0] for d, c in zip(cam_data, color_data)
        ] if animated else None

        return graph_axis, graph_target, graph_follower, graph_vert, graph_follower_pnt

    @staticmethod
    def _update_subplot_graph(graph_vert, graph_follower_pnt, theta, frame, indices, data_view):
        theta_deg = to_degrees(theta)
        graph_vert.set_xdata(2 * [theta_deg])

        for d, i in zip(data_view, indices):
            graph_follower_pnt[i].set_xdata(theta_deg)
            graph_follower_pnt[i].set_ydata(d[frame])

        return graph_vert, graph_follower_pnt

    def plot(self, data, size=5, animated=False, frames=100, show=True, save=False, filename=None):
        if filename is None:
            filename = './fig.gif' if animated else './fig.png'

        fig = plt.figure(figsize=(2.4 * size, size))
        gs = gds.GridSpec(3, 2)
        ax_cam = fig.add_subplot(gs[:, 0])
        ax_graph_pos = fig.add_subplot(gs[0, 1])
        ax_graph_vel = fig.add_subplot(gs[1, 1], sharex=ax_graph_pos)
        ax_graph_acc = fig.add_subplot(gs[2, 1], sharex=ax_graph_pos)

        cmap = mpl.colormaps['hsv']

        theta = np.linspace(0., 2. * np.pi, self.points)
        angle_view = np.linspace(0., 2. * np.pi, frames)

        data_complete = [{
            'label': key,
            'index': index,
            'color': cmap(linear(index, 0, 0., len(data), 1.)),
            'points': value,
            'displacement': self.displacement(value, self.points),
            'displacement_view': None,
            'velocity': None,
            'velocity_view': None,
            'acceleration': None,
            'acceleration_view': None,
        } for index, (key, value) in enumerate(data.items())
        ]

        for d in data_complete:
            d['velocity'] = np.gradient(d['displacement'])
            d['acceleration'] = np.gradient(d['velocity'])

            if animated:
                d['displacement_view'] = np.interp(angle_view, theta, d['displacement'])
                d['velocity_view'] = np.interp(angle_view, theta, d['velocity'])
                d['acceleration_view'] = np.interp(angle_view, theta, d['acceleration'])

        displacement = np.array([self._dist + self._path(t) for t in theta])
        velocity = np.gradient(displacement)
        acceleration = np.gradient(velocity)

        radius_min = np.min([np.min(d['points']) for d in data_complete])
        radius_max = np.max([np.max(d['points']) for d in data_complete])
        x_span = np.array([-radius_max, radius_max])

        cam_origin = ax_cam.plot(0., 0., linestyle='', marker='o', color='black')[0]

        cam_circ = [
            ax_cam.plot(
                *to_cartesian(theta, np.full_like(theta, np.max(d['points']))),
                linestyle=':', color=d['color']
            )[0] for d in data_complete
        ]

        cam_draw = [
            ax_cam.plot(
                *to_cartesian(theta, d['points']),
                color=d['color'], label=d['label']
            )[0] for d in data_complete
        ]

        cam_target = ax_cam.plot(
            x_span, np.full_like(x_span, displacement[0]),
            linestyle=':', color='black', label='Target'
        )[0] if animated else None

        cam_follower = [
            ax_cam.plot(
                x_span, np.full_like(x_span, d['displacement'][0]),
                linestyle='--', color=d['color']
            )[0] for d in data_complete
        ] if animated else None

        graph_pos_axis, graph_pos_target, graph_pos_follower, graph_pos_vert, graph_pos_follower_pnt = \
            self._build_subplot_graph(
                ax_graph_pos,
                theta,
                self._dist,
                displacement,
                [d['displacement'] for d in data_complete],
                [d['color'] for d in data_complete],
                animated=animated,
                theta_view=angle_view,
            )

        graph_vel_axis, graph_vel_target, graph_vel_follower, graph_vel_vert, graph_vel_follower_pnt = \
            self._build_subplot_graph(
                ax_graph_vel,
                theta,
                0.,
                velocity,
                [d['velocity'] for d in data_complete],
                [d['color'] for d in data_complete],
                animated=animated,
                theta_view=angle_view,
            )

        graph_acc_axis, graph_acc_target, graph_acc_follower, graph_acc_vert, graph_acc_follower_pnt = \
            self._build_subplot_graph(
                ax_graph_acc,
                theta,
                0.,
                acceleration,
                [d['acceleration'] for d in data_complete],
                [d['color'] for d in data_complete],
                animated=animated,
                theta_view=angle_view,
            )

        def update(frame):
            angle = theta + 2. * np.pi * frame / frames
            angle0 = angle[0]
            angle0_deg = to_degrees(angle0)

            cam_target.set_ydata(np.full_like(x_span, self._dist + self.path(angle0)))

            graph_pos_vert.set_xdata(2 * [angle0_deg])
            graph_vel_vert.set_xdata(2 * [angle0_deg])
            graph_acc_vert.set_xdata(2 * [angle0_deg])

            # indices = [d['index'] for d in data_complete]
            # self._update_subplot_graph(graph_pos_vert, graph_pos_follower_pnt, angle, frame, indices, [d['displacement_view'] for d in data_complete])
            # self._update_subplot_graph(graph_vel_vert, graph_vel_follower_pnt, angle, frame, indices, [d['velocity_view'] for d in data_complete])
            # self._update_subplot_graph(graph_acc_vert, graph_acc_follower_pnt, angle, frame, indices, [d['acceleration_view'] for d in data_complete])

            for d in data_complete:
                index = d['index']
                x_cam, y_cam = to_cartesian(angle, d['points'])

                cam_draw[index].set_xdata(x_cam)
                cam_draw[index].set_ydata(y_cam)

                cam_follower[index].set_ydata(np.full_like(x_span, d['displacement_view'][frame]))

                graph_pos_follower_pnt[index].set_xdata(angle0_deg)
                graph_pos_follower_pnt[index].set_ydata(d['displacement_view'][frame])

                graph_vel_follower_pnt[index].set_xdata(angle0_deg)
                graph_vel_follower_pnt[index].set_ydata(d['velocity_view'][frame])

                graph_acc_follower_pnt[index].set_xdata(angle0_deg)
                graph_acc_follower_pnt[index].set_ydata(d['acceleration_view'][frame])

            return (
                cam_draw, cam_follower, cam_target,
                graph_pos_vert, graph_pos_follower_pnt,
                graph_vel_vert, graph_vel_follower_pnt,
                graph_acc_vert, graph_acc_follower_pnt,
            )

        fig.legend(loc='upper right')
        ax_cam.set_title('Cam-Follower')
        self._config_subplot(ax_graph_pos, 'Position')
        self._config_subplot(ax_graph_vel, 'Velocity')
        self._config_subplot(ax_graph_acc, 'Acceleration')

        if animated:
            ani = anm.FuncAnimation(fig=fig, func=update, frames=frames, interval=1)

        if show:
            plt.show()

        if save:
            if animated:
                ani.save(filename=f"{filename}.gif", writer="pillow")
            else:
                fig.savefig(filename)

    def find_max_size(self, r, n=1000):
        data = np.full(self.points, r)
        theta = np.linspace(0., 2. * np.pi, self.points)
        phase = np.linspace(0., 2. * np.pi, n)

        for i in range(n):
            for j in range(self.points):
                x, y = to_cartesian(theta[j] + phase[i], data[j])
                dist = self._dist + self._path(phase[i])
                if y > dist:
                    data[j] = dist / np.abs(np.sin(theta[j] + phase[i]))

        self._data_max = data
        return self._data_max

    def find_min_size(self, div=10, interp=None, maxiter=100, verbose=False):
        if interp is None:
            interp = lambda x, y: CubicSpline(x, y, bc_type='natural')

        theta = np.linspace(0., 2. * np.pi, int(self.points / div))
        theta_aug = np.linspace(0., 2. * np.pi, self.points)
        indices = np.array([arg_closest(theta_aug, t) for t in theta])

        def minimize(x):
            x[-1] = x[0]
            interpolator = interp(theta, x)
            data_aug = interpolator(theta_aug)

            min1 = np.sqrt(np.sum(self.verify(data_aug, self.points) ** 2.) / self.points)
            min2 = np.sqrt(np.sum(x ** 2.) / len(x))
            return min1 + 5e-3 * min2

        bounds = [(0.2 * self._data_max[i], self._data_max[i]) for i in indices]
        res = opt.minimize(
            minimize, x0=self._data_max[indices], bounds=bounds,
            method='trust-constr', options={'maxiter': maxiter, 'verbose': 2 if verbose else 0}
        )

        resx = res.x
        resx[-1] = resx[0]
        cs = CubicSpline(theta, resx, bc_type='periodic')
        self._data_min = cs(theta_aug)
        return self._data_min

    def optimize(self, minimize, constraints=None, div=10, interp=None, maxiter=100, verbose=False):
        if interp is None:
            interp = lambda x, y: CubicSpline(x, y, bc_type='natural')

        theta = np.linspace(0., 2. * np.pi, int(self.points / div))
        theta_aug = np.linspace(0., 2. * np.pi, self.points)
        indices = np.array([arg_closest(theta_aug, t) for t in theta])

        constr = [opt.NonlinearConstraint(lambda x: c['fun'](x, theta, theta_aug), c['lb'], c['ub'])
                  for c in constraints]

        bounds = [(0.2 * self._data_max[i], self._data_max[i]) for i in indices]
        res = opt.minimize(
            minimize, args=(theta, theta_aug),
            x0=self._data_max[indices], bounds=bounds, constraints=constr,
            method='trust-constr', options={'maxiter': maxiter, 'verbose': 2 if verbose else 0}
        )

        resx = res.x
        resx[-1] = resx[0]
        interpolator = interp(theta, resx)
        self._data_min = interpolator(theta_aug)
        return self._data_min

    def displacement(self, data, n=1000):
        theta = np.linspace(0., 2. * np.pi, self.points)
        phase = np.linspace(0., 2. * np.pi, n)
        position = np.full_like(phase, -np.inf, dtype=np.float64)

        for i in range(n):
            y_max = -np.inf
            for j in range(self.points):
                x, y = to_cartesian(theta[j] + phase[i], data[j])
                if y > y_max:
                    y_max = y

            if self._constrained:
                position[i] = max(y_max, self._dist)
            else:
                position[i] = y_max

        return position

    def verify(self, data, n=1000):
        position = self.displacement(data, n)
        phase = np.linspace(0., 2. * np.pi, n)
        residuals = np.full_like(phase, np.inf, dtype=np.float64)

        for i in range(n):
            dist = self._dist + self._path(phase[i])
            residuals[i] = np.abs(position[i] - dist)

        return residuals

    def project(self, r, n=1000):
        self._data_min = self.find_max_size(r, n)
